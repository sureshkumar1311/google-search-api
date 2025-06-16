from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel, field_validator, ValidationError, ConfigDict
from openai import AzureOpenAI
from langchain.tools import BaseTool
import azure.functions as func
import aiohttp  
import time
import asyncio
import json
from typing import Optional, List, Dict, Any, Tuple  
from datetime import datetime
import uuid
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry._logs import get_logger_provider, set_logger_provider
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import os
import backoff
import openai
import contextvars
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Context variable for correlation ID
correlation_id_context = contextvars.ContextVar('correlation_id')

# Set up OpenTelemetry Logger Provider and Azure Monitor exporter
set_logger_provider(LoggerProvider())
exporter = AzureMonitorLogExporter(
    connection_string=os.environ.get("APP_INSIGHTS_CONN_STRING")
)
get_logger_provider().add_log_record_processor(BatchLogRecordProcessor(exporter))

# Configure the root logger
logging.basicConfig(level=logging.DEBUG)
handler = LoggingHandler()
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(handler)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

class CorrelationFormatter(logging.Formatter):
    def converter(self, timestamp):
        return time.gmtime(timestamp)

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return s

    def format(self, record):
        # Get correlation ID from context or generate new one
        try:
            correlation_id = correlation_id_context.get()
        except LookupError:
            correlation_id = str(uuid.uuid4())
            
        record.correlation_id = getattr(record, 'correlation_id', None) or correlation_id
        record.user_name = getattr(record, 'user_name', 'N/A')
        record.action_field = getattr(record, 'action_field', 'N/A')
        record.response_time = getattr(record, 'response_time', 'N/A')
        return super().format(record)

# Add formatter to the handler
formatter = CorrelationFormatter(
    '%(asctime)s %(name)s %(levelname)s [%(correlation_id)s] [user_name=%(user_name)s] '
    '[action_field=%(action_field)s] [response_time=%(response_time)s]: %(message)s'
)
handler.setFormatter(formatter)

class CustomLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})
        try:
            extra['correlation_id'] = self.extra.get('correlation_id', correlation_id_context.get())
        except LookupError:
            extra['correlation_id'] = str(uuid.uuid4())
        kwargs['extra'] = extra
        return msg, kwargs

# Create logger adapter
logger = CustomLoggerAdapter(root_logger, {})

###################################################################################################

def validate_environment_variables():
    """Validate required environment variables at startup"""
    required_vars = [
        'GPT_API_VERSION', 'GPT_API_KEY', 'GPT_BASE_URL', 'Model',
        'GoogleAPIKey', 'GoogleSearchEngineID', 'APP_INSIGHTS_CONN_STRING',
        'CosmosDBSave', 'CosmosErrorLog'
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate environment variables at startup
try:
    validate_environment_variables()
except ValueError as e:
    logger.error(f"Environment validation failed: {e}")
    raise

# Pydantic settings
class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    api_version: str = Field(..., alias='GPT_API_VERSION')
    api_key: str = Field(..., alias='GPT_API_KEY')
    base_url: str = Field(..., alias='GPT_BASE_URL')
    model: str = Field(..., alias='Model')
    google_api_key: str = Field(..., alias='GoogleAPIKey')
    google_search_engine_id: str = Field(..., alias='GoogleSearchEngineID')

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    function_call: Optional[Dict] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatHistory(BaseModel):
    messages: List[Message]
    name: Optional[str] = None
    question: Optional[str] = None


class MasterSearchParams(BaseModel):
    operation: str
    keywords: str
    region: Optional[str] = 'wt-wt'
    max_results: Optional[int] = 5
    to_lang: Optional[str] = 'en'
    place: Optional[str] = None

    @field_validator('operation')
    def validate_operation(cls, v):
        allowed_operations = {'text', 'image', 'video', 'news', 'suggestions'}
        if v not in allowed_operations:
            raise ValueError(
                f"Invalid operation '{v}'. Allowed operations are: {allowed_operations}")
        return v
    
    
    @field_validator('keywords')
    def validate_keywords_length(cls, v):
        if len(v) > 50:
            raise ValueError("Keywords must not exceed 50 characters")
        return v

#########################################################################################################################

class CustomException(Exception):
    def __init__(self, message, error_code, action_field=None):
        super().__init__(message)
        self.error_code = error_code
        self.action_field = action_field

def log_error_and_raise(message: str, error_code: int, action_field: str, exception: Exception = None):
    """Helper function to log errors and raise CustomException"""
    logger.exception(f"{message}", exc_info=True, extra={
        "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
        "action_field": action_field
    })
    raise CustomException(message, error_code, action_field) from exception

######################################################


async def save_document(item: Dict[str, Any]) -> Dict[str, Any]:
    """Save document to database using async HTTP client"""
    db_url = os.environ.get('CosmosDBSave')
    req_body = item

    try:
        async with aiohttp.ClientSession() as session:
            logger.debug(
                f"Request initiated to save the document to the database for the id {req_body.get('item_id')}.",
                extra={
                    "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                    "action_field": "Save Document Helper API."
                }
            )
            
            async with session.post(db_url, json=req_body) as response:
                response.raise_for_status()
                result = await response.json()

                logger.info(
                    "Request was successfully saved to the database. Returning final response.",
                    extra={
                        "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                        "action_field": "Save Document Helper API."
                    }
                )
                return result

    except aiohttp.ClientError as e:
        log_error_and_raise(
            f"An error occurred saving the document to db: {str(e)}", 
            500, "SaveDocumentAPI", e
        )
    except Exception as e:
        log_error_and_raise(
            f"Unexpected error in save_document: {str(e)}", 
            500, "SaveDocumentAPI", e
        )


async def error_logging(user_name: str, error_message: str, user_question: str,
                       assistant_name: str = 'Knowledge', error_status: int = 500,
                       error_description: str = 'UnexpectedError', action_field: str = None) -> Dict[str, Any]:
    """Log errors to database using async HTTP client"""
    error_log_url = os.environ.get('CosmosErrorLog')
    req_body = {
        'UserName': user_name,
        'AssistantName': assistant_name,
        'ErrorMessage': error_message,
        'ErrorStatus': error_status,
        'ErrorDescription': error_description,
        'Question': user_question,
        'ActionField': action_field
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(error_log_url, json=req_body) as response:
                response.raise_for_status()
                result = await response.json()
                
                logger.info(
                    "Error Log was successfully saved to the database. Returning final response.",
                    extra={
                        "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                        "action_field": "Error Logging Helper API"
                    }
                )
                return result

    except aiohttp.ClientError as e:
        log_error_and_raise(
            f"Failed to save error log: {str(e)}", 
            500, "ErrorLoggingAPI", e
        )
    except Exception as e:
        log_error_and_raise(
            f"Unexpected error in error_logging: {str(e)}", 
            500, "ErrorLoggingAPI", e
        )

# Async function to make Google Search API request
async def make_search_request_async(query: str, max_results: int = 8) -> dict:
    """
    Wraps the blocking googleapiclient call in asyncio.to_thread for non-blocking behavior.
    """
    def sync_search():
        """
        This inner function does the actual synchronous call.
        """
        try:
            service = build("customsearch", "v1", developerKey=os.environ.get("GoogleAPIKey"))
            search_engine_id = os.environ.get("GoogleSearchEngineID")
            return service.cse().list(
                q=query,
                cx=search_engine_id,
                num=min(max_results, 10)
            ).execute()
        except HttpError as e:
            if e.resp.status in [429, 403]:  # Re-raise for retry if needed
                raise
            else:
                # Non-retriable HTTP error
                raise

    # Offload the synchronous call to a separate thread
    result = await asyncio.to_thread(sync_search)
    return result

# Google Search Tool with async support
class GoogleSearchTool(BaseTool):
    name: str = "Intermediate Answer"
    description: str = "useful for when you need to answer questions about current events and dates"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((HttpError,))
    )
    async def run_async(self, query: str, max_results: int = 8) -> str:
        """
        Asynchronous replacement for the synchronous _run method.
        Retries the google search up to 3 times on HttpError 429 or 403.
        """
        try:
            result = await make_search_request_async(query, max_results)
            result_string = ""
            if 'items' in result:
                for index, item in enumerate(result['items'], start=1):
                    if index <= max_results:
                        result_string += (
                            f"{index}. Topic: {item['title']} \n"
                            f"  Content: {item['snippet']}\n"
                            f"  URL: {item['link']}\n\n"
                        )
            logger.info("GoogleSearchTool used successfully.")
            return result_string
        except Exception as ex:
            logger.exception(f"An error occurred in GoogleSearchTool: {ex}")
            raise

    def _run(self, query: str) -> str:
        """
        Synchronous method for backward compatibility.
        """
        try:
            service = build("customsearch", "v1", developerKey=os.environ.get("GoogleAPIKey"))
            search_engine_id = os.environ.get("GoogleSearchEngineID")
            
            result = service.cse().list(
                q=query,
                cx=search_engine_id,
                num=8
            ).execute()

            result_string = ""
            if 'items' in result:
                for index, item in enumerate(result['items'], start=1):
                    if index < 9:
                        result_string += f"{index}. Topic: {item['title']} \n  Content: {item['snippet']}\n  URL: {item['link']}\n\n"
                    else:
                        break

            logger.info("GoogleSearchTool used successfully.",
                        extra={
                            "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                            "action_field": "GoogleSearchTool"
                        })
            return result_string

        except HttpError as e:
            if e.resp.status in [429, 403]:  # Rate limit or quota exceeded
                logger.warning(
                    f"Google Search API rate limit/quota error: {e.resp.status}. Retrying...",
                    extra={
                        "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                        "action_field": "GoogleSearchTool"
                    }
                )
                raise  # Re-raise to trigger retry
            else:
                # For other HTTP errors, don't retry
                raise CustomException(
                    f"Google Search API error: {str(e)}", 
                    error_code=e.resp.status, 
                    action_field="GoogleSearchToolAPI"
                ) from e
        except Exception as ex:
            log_error_and_raise(
                f"An error occurred in GoogleSearchTool: {str(ex)}", 
                500, "GoogleSearchToolAPI", ex
            )

    #Properly implement the async _arun method to avoid inconsistency
    async def _arun(self, query: str) -> str:
        """
        Properly implement the async _arun method by calling run_async.
        This ensures consistency and allows the tool to work in async-only contexts.
        """
        return await self.run_async(query)

async def master_search(input_json: str) -> Dict[str, str]:
    """
    Master search function that returns a dictionary.
    Handle different operations and use all parameters from MasterSearchParams.
    """
    try:
        googlesearch = GoogleSearchTool()
        params = MasterSearchParams(**json.loads(input_json))

        # Handle different operations here (text vs. image vs. others).
        # Use all parameters including operation, max_results, etc.
        search_query = params.keywords
        
        if params.operation == "text":
            # Standard text search
            result = await googlesearch.run_async(search_query, params.max_results or 8)
        elif params.operation == "image":
            # Image search - modify query to include image search context
            result = await googlesearch.run_async(f"{search_query} images", params.max_results or 8)
        elif params.operation == "video":
            # Video search - modify query to include video search context
            result = await googlesearch.run_async(f"{search_query} videos", params.max_results or 8)
        elif params.operation == "news":
            # News search - modify query to include news search context
            result = await googlesearch.run_async(f"{search_query} news", params.max_results or 8)
        elif params.operation == "suggestions":
            # Suggestions search - modify query to get related suggestions
            result = await googlesearch.run_async(f"{search_query} related suggestions", params.max_results or 8)
        else:
            # Fallback for any other operations
            result = await googlesearch.run_async(f"{search_query} {params.operation}", params.max_results or 8)

        logger.info(
            f"master_search actions done successfully for operation: {params.operation}",
            extra={
                "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                "action_field": "GoogleSearchMasterSearch"
            }
        )
        return {"results": result}
        
    except ValidationError as ex:
        log_error_and_raise(
            f"Validation error in 'master_search': {str(ex)}", 
            400, "GoogleSearchMasterSearch", ex
        )
    except Exception as ex:
        log_error_and_raise(
            f"Unexpected error in 'master_search': {str(ex)}", 
            500, "GoogleSearchMasterSearch", ex
        )

async def async_master_search(input_json):
    result = await master_search(input_json)
    return result["results"]  # Extract results string for backward compatibility

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "master_search",
            "description": "Utilizes multiple search operations (text, image, video, news,suggestions) to retrieve relevant information based on the specified operation and keywords. This tool is designed to provide diverse search capabilities using a single function.Use this function when user asks only in english language",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["text", "image", "video", "news", "suggestions"],
                        "description": "The type of search operation to perform. Options include 'text', 'image', 'video', 'news', 'suggestions'."
                    },
                    "keywords": {
                        "type": "string",
                        "description": "The keywords to search for in the specified operation.Maximum keyword should only be 50 characters"
                    },
                    "region": {
                        "type": "string",
                        "description": "Optional. The region to perform the search in. Defaults to 'wt-wt'."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Optional. The maximum number of results to return. If not provided, all results will be returned."
                    }
                },
                "required": [
                    "operation",
                    "keywords"
                ]
            }
        }
    }
]

##########################################################################################################################

@backoff.on_exception(backoff.expo,
                      (openai.APIConnectionError, openai.RateLimitError),
                      max_time=180,
                      logger=logger)
def make_openai_api_call(client, messages, settings, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, tools):
    response = client.chat.completions.create(
        messages=messages,
        model=settings.model,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.model_dump()

def validate_chat_history(input_data):
    """Helper function to validate chat history"""
    try:
        if isinstance(input_data, ChatHistory):
            messages = input_data.messages
        elif isinstance(input_data, list):
            messages = input_data
        else:
            raise ValueError("Invalid Data Format. Expected ChatHistory or list of Messages object.")
        return messages
    except ValidationError as e:
        log_error_and_raise("Invalid ChatHistory data.", 400, "GetCompletionCall", e)
    except Exception as e:
        log_error_and_raise("Invalid ChatHistory data.", 400, "GetCompletionCall", e)

def create_openai_client():
    """Helper function to create OpenAI client"""
    try:
        settings = Settings()
        client = AzureOpenAI(
            api_version=settings.api_version,
            api_key=settings.api_key,
            base_url=settings.base_url
        )
        logger.info("AzureOpenAI client defined.",
                    extra={
                        "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                        "action_field": "GetCompletionCall",
                    })
        return client, settings
    except ValidationError as e:
        log_error_and_raise("OpenAI Configuration validation error.", 500, "GetCompletionCall", e)

def get_completion(input_data, tools, temperature=0, max_tokens=4095, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    messages = validate_chat_history(input_data)
    client, settings = create_openai_client()

    validated_chat_history = ChatHistory(messages=messages)
    messages = validated_chat_history.model_dump()['messages'] or []

    logger.info("Sending completion request to OpenAI API.",
                extra={
                    "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                    "action_field": "GetCompletionCall",
                })
    
    logger.debug(f"Messages: {messages}",
                 extra={
                     "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                     "action_field": "GetCompletionCall",
                 })
    
    logger.debug(f"Functions: {tools}",
                 extra={
                     "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                     "action_field": "GetCompletionCall",
                 })

    try:
        start_time = time.time()
        response = make_openai_api_call(
            client, messages, settings, temperature, max_tokens, top_p, 
            frequency_penalty, presence_penalty, stop, tools=tools
        )
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        logger.info("Received response from OpenAI API.",
                    extra={
                        "correlation_id": correlation_id_context.get(str(uuid.uuid4())),
                        "action_field": "OpenAIAPICall",
                        "response_time": response_time_ms
                    })
        return response['choices'][0]['message'], response['usage']

    except CustomException:
        raise
    except openai.APIConnectionError as e:
        log_error_and_raise(f"Failed to connect to OpenAI API. {str(e)}", 503, "OpenAIAPICall", e)
    except openai.AuthenticationError as e:
        log_error_and_raise(f"Authentication failed.{str(e)}", 401, "GetCompletionCall", e)
    except openai.RateLimitError as e:
        log_error_and_raise(f"API rate limit exceeded. {str(e)}", 429, "OpenAIAPICall", e)
    except openai.APITimeoutError as e:
        log_error_and_raise(f"Request timed out. {str(e)}", 504, "OpenAIAPICall", e)
    except openai.BadRequestError as e:
        log_error_and_raise(f"Bad request, please check parameters. {str(e)}", 400, "OpenAIAPICall", e)
    except openai.ConflictError as e:
        log_error_and_raise(f"Resource conflict error. {str(e)}", 409, "OpenAIAPICall", e)
    except openai.InternalServerError as e:
        log_error_and_raise(f"OpenAI server error. {str(e)}", 500, "OpenAIAPICall", e)
    except openai.NotFoundError as e:
        log_error_and_raise(f"Requested resource not found. {str(e)}", 404, "OpenAIAPICall", e)
    except openai.PermissionDeniedError as e:
        log_error_and_raise(f"Permission denied. {str(e)}", 403, "OpenAIAPICall", e)
    except Exception as e:
        log_error_and_raise(f"Unexpected error occurred while using the OpenAI API. {str(e)}", 500, "OpenAIAPICall", e)

##########################################################################################################################

async def process_tool_calls(response, message):
    """Helper function to process tool calls"""
    for tool_call in response['tool_calls']:
        tool_call_id = tool_call['id']
        tool_name = tool_call['function']['name']
        function_response = ""

        try:
            if tool_name == "master_search":
                arguments = json.loads(tool_call['function']['arguments'])
                # Use await instead of asyncio.run()
                function_response = await async_master_search(json.dumps(arguments))

            message.messages.append(
                Message(
                    tool_call_id=tool_call_id,
                    role="assistant",
                    name=tool_name,
                    content=function_response
                )
            )
        except Exception as e:
            logger.error(f"Error processing tool call {tool_call_id}: {e}")
            raise


async def get_answer(chat_history: ChatHistory) -> Tuple[str, Dict[str, Any]]:
    """Get answer from the AI model with proper tool call handling"""
    message = chat_history.model_copy()
    
    try:
        response, usage = get_completion(input_data=message, tools=tools)
    except CustomException:
        raise

    MAX_LOOP_ITERATIONS = 15
    iteration_counter = 0

    while True:
        iteration_counter += 1
        if iteration_counter > MAX_LOOP_ITERATIONS:
            log_error_and_raise(
                "Exceeded maximum iterations in tool call processing loop.",
                500, "FunctionToolCall"
            )

        if response.get('tool_calls'):
            # Process tool calls
            await process_tool_calls(response, message)
            # Get next response after processing tool calls
            next_response, usage = get_completion(input_data=message, tools=tools)
            response = next_response
            # Continue the loop instead of returning immediately
            # This allows for multiple consecutive tool calls
        else:
            message.messages.append(
                Message(
                    role="assistant",
                    content=response.get('content', ""),
                )
            )
            return response.get('content', ""), usage

##########################################################################################################################

def create_error_response(message: str, status_code: int, error_code: int = None, action_field: str = None):
    """Helper function to create standardized error responses"""
    error_response = {
        "role": "assistant",
        "content": message,
        "error": True,
        "errorAt": datetime.utcnow().isoformat() + "Z"
    }
    
    if error_code:
        error_response["error_code"] = error_code
    if action_field:
        error_response["action_field"] = action_field
        
    return func.HttpResponse(
        body=json.dumps(error_response),
        mimetype="application/json",
        status_code=status_code
    )

def validate_request_body(req: func.HttpRequest):
    """Helper function to validate request body"""
    chat_history_json = req.get_json()
    if not chat_history_json:
        raise ValueError("Request body is empty")
    
    if 'request_message' not in chat_history_json:
        raise ValueError("Missing 'request_message' in request body")
    
    if 'user_details' not in chat_history_json:
        raise ValueError("Missing 'user_details' in request body")
    
    return chat_history_json

@app.route(route="knowledgeAgent")
async def knowledgeAgent(req: func.HttpRequest) -> func.HttpResponse:
    # Generate correlation ID for this request
    request_correlation_id = str(uuid.uuid4())
    correlation_id_context.set(request_correlation_id)
    
    logger.info(
        'Python HTTP trigger function processed a request.',
        extra={
            "correlation_id": request_correlation_id,
            "action_field": "MainCall",
        }
    )

    try:
        # Validate request body
        chat_history_json = validate_request_body(req)
        
        chat_history_dict = {"messages": chat_history_json['request_message']}
        user_question = chat_history_json['request_message'][-1]['content']
        user_Details = chat_history_json['user_details']
        user_name = user_Details.get('username', 'Unknown')

        # Validate chat history
        chat_history = ChatHistory(**chat_history_dict)

        # Get response from model - use await since get_answer is now async
        ans, usage = await get_answer(chat_history)
        
        # Validate response
        if not ans:
            raise CustomException("Received null response from model", error_code=500)

        # Prepare response data
        response_data = {
            "role": "assistant",
            "content": ans,
            "usage": usage
        }

        # Update user details
        user_Details['answer'] = ans
        user_Details['question'] = user_question
        user_Details['usage'] = usage

        # Log successful response
        logger.info(
            "Successfully processed request",
            extra={
                "correlation_id": request_correlation_id,
                "user_name": user_name,
                "action_field": "MainCall",
                "response_size": len(ans) if ans else 0
            }
        )

        return func.HttpResponse(
            body=json.dumps(response_data),
            mimetype="application/json",
            status_code=200
        )

    except ValueError as e:
        return create_error_response(str(e), 400)

    except CustomException as e:
        return create_error_response(
            str(e), e.error_code, e.error_code, e.action_field
        )

    except Exception as e:
        logger.exception(
            f"Unexpected error in knowledgeAgent: {str(e)}",
            extra={
                "correlation_id": request_correlation_id,
                "action_field": "MainCall"
            }
        )
        return create_error_response(
            f"An unexpected error occurred: {str(e)}", 500
        )
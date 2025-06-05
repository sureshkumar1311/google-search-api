from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel, field_validator, ValidationError, ConfigDict
from openai import AzureOpenAI
from langchain.tools import BaseTool
import azure.functions as func
import requests
import time
import asyncio
import json
from typing import Optional
import logging
import os
import backoff
import openai
from typing import Optional, List, Dict
from datetime import datetime
import uuid
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LoggingHandler,
)
from opentelemetry._logs import (
    get_logger_provider,
    set_logger_provider,
)

# Set up OpenTelemetry Logger Provider and Azure Monitor exporter
set_logger_provider(LoggerProvider())
exporter = AzureMonitorLogExporter(
    connection_string=os.environ.get("APP_INSIGHTS_CONN_STRING")
)
get_logger_provider().add_log_record_processor(BatchLogRecordProcessor(exporter))

# Configure the root logger directly using basicConfig (better for Azure Functions)
logging.basicConfig(level=logging.DEBUG)
handler = LoggingHandler()  # This handler sends logs to Azure Monitor
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(handler)

# Create a correlation ID
correlation_id = str(uuid.uuid4())

# Custom formatter to include correlation ID and optional fields

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


class CorrelationFormatter(logging.Formatter):
    def converter(self, timestamp):
        return time.gmtime(timestamp)  # Use UTC time

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return s

    def format(self, record):
        # Ensure the correlation ID is added if it isn't already set
        record.correlation_id = getattr(
            record, 'correlation_id', None) or correlation_id

        # Handle optional fields dynamically
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

# Custom LoggerAdapter to ensure extra fields are passed


class CustomLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Merge extra fields into the log record
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = self.extra.get(
            'correlation_id', correlation_id)
        kwargs['extra'] = extra
        return msg, kwargs


# Create a logger adapter that includes the correlation ID
logger = CustomLoggerAdapter(root_logger, {"correlation_id": correlation_id})

###################################################################################################

# pydantic settings


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
    subscription_key: str = Field(..., alias='BingKey')
    endpoint: str = Field(..., alias='BingEndPoint')


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
        allowed_operations = {'text', 'image', 'video',
                              'news', 'suggestions'}
        if v not in allowed_operations:
            raise ValueError(
                f"Invalid operation '{v}'. Allowed operations are: {allowed_operations}")
        return v

#########################################################################################################################
# custom exception class


class CustomException(Exception):
    def __init__(self, message, error_code, action_field=None):
        super().__init__(message)
        self.error_code = error_code
        self.action_field = action_field


######################################################

def save_document(item: json):
    db_url = os.environ.get('CosmosDBSave')

    # Extracting required fields from item
    req_body = item

    try:
        # Sending the POST request
        response = requests.post(db_url, json=req_body)
        logger.debug(
            f"Request initiated to save the document to the database for the id {req_body.get('item_id')}.",
            extra={
                "correlation_id": correlation_id,
                "action_field": "Save Document Helper API."
            }
        )
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Return the response JSON data if successful
        logger.info(
            "Request was successfully saved to the database. Returning final response.",
            extra={
                "correlation_id": correlation_id,
                "action_field": "Save Document Helper API."
            }
        )
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.exception(
            f"An error occurred saving the document to db.", exc_info=True,
            extra={
                "correlation_id": correlation_id,
                "action_field": "Save Document Helper API."
            }
        )
        raise CustomException(
            f"Failed to save document: {str(e)}", error_code=500, action_field="SaveDocumentAPI")


def error_logging(user_name: str, error_message: str, user_question: str,
                  assistant_name: str = 'Knowledge', error_status: int = 500,
                  error_description: str = 'UnexpectedError', action_field=None):

    error_log_url = os.environ.get('CosmosErrorLog')

    # Constructing the JSON payload within the function
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
        # Sending the POST request
        response = requests.post(error_log_url, json=req_body)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()
        logger.info(
            "Error Log was successfully saved to the database. Returning final response.",
            extra={

                "correlation_id": correlation_id,
                "action_field": "Error Logging Helper API"
            }
        )
        # Return the response JSON data if successful
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.exception(
            f"An error occurred saving the document to db.", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "action_field": "Error Logging Helper API"
            }
        )
        # Raise an exception with details if the request fails
        raise CustomException(
            f"Failed to save document:{str(e)}", error_code=500, action_field="ErrorLoggingAPI")


class BingSearchTool(BaseTool):
    name: str = "Intermediate Answer"
    description: str = "useful for when you need to answer questions about current events and dates"

    def _run(self, query: str) -> str:
        # os.environ.get("BingKey")
        subscription_key = os.environ.get("BingKey")
        
        endpoint = "https://api.bing.microsoft.com/v7.0/search"
        mkt = 'en-US'
        params = {'q': query, 'mkt': mkt}
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}

        try:
            response = requests.get(endpoint, headers=headers, params=params)
            logger.info(
                "API request to the BingSearchTool made.",
                extra={

                    "correlation_id": correlation_id,
                    "action_field": "BingSearchTool"
                }
            )
            logger.debug(
                f"BingSearchTool response status--{str(response.status_code)}", exc_info=True,
                extra={

                    "correlation_id": correlation_id,
                    "action_field": "BingSearchTool"
                }
            )
            response.raise_for_status()

            search_results = response.json().get('webPages', {}).get('value', [])
            result_string = ""
            for index, result in enumerate(search_results, start=1):
                if index < 9:
                    result_string += f"{index}. Topic: {result['name']} \n  Content: {result['snippet']}\n URL: {result['url']}\n\n"
                else:
                    break
            logger.info("BingSearchTool used successfully.",
                        extra={

                            "correlation_id": correlation_id,
                            "action_field": "BingSearchTool"
                        }
                        )
            return result_string

        except Exception as ex:
            logger.exception(
                f"Encountered an exception in BingSearchTool.", exc_info=True,
                extra={

                    "correlation_id": correlation_id,
                    "action_field": "BingSearchToolAPI"
                }
            )
            raise CustomException(
                f"An error occurred in BingSearchT: {str(ex)}", error_code=500, action_field="BingSearchToolAPI") from ex

    def _arun(self, query: str) -> str:
        logger.error("BingSearchTool does not support async processing.", exc_info=True,
                     extra={

                         "correlation_id": correlation_id,
                         "action_field": "BingSearchToolRun"
                     }
                     )
        raise CustomException(
            "'NotImplementedError': BingSearchTool does not support async.", error_code=501, action_field="BingSearchToolRun")


async def master_search(input_json: str) -> dict:
    try:
        bingsearch = BingSearchTool()
        # Validate input parameters using MasterSearchParams
        params = MasterSearchParams(**json.loads(input_json))

        search_actions = {
            'text': lambda: bingsearch.run(params.keywords),
            'image': lambda: bingsearch.run(params.keywords),
            'video': lambda: bingsearch.run(params.keywords),
            'news': lambda: bingsearch.run(params.keywords),
            # 'map': lambda: bingsearch.run(params.keywords),
            # 'translate': lambda: bingsearch.run(params.keywords),
            'suggestions': lambda: bingsearch.run(params.keywords)
        }

        if params.operation in search_actions:
            logger.info(
                "master_search actions done successfully.",
                extra={

                    "correlation_id": correlation_id,
                    "action_field": "BingSearchMasterSearch"
                }
            )
            return search_actions[params.operation]()
        else:
            logger.error(
                "Invalid operation in 'master_search'. Please choose from: 'text', 'image', 'video', 'news', 'suggestions'.", exc_info=True,
                extra={

                    "correlation_id": correlation_id,
                    "action_field": "BingSearchMasterSearch"
                }
            )
            raise CustomException(
                "Invalid operation in 'master_search'. Choose from: 'text', 'image', 'video', 'news', 'suggestions'.", error_code=400, action_field="BingSearchMasterSearch")

    except ValueError as e:
        logger.exception(
            f"Validation error in 'master_search'.", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "action_field": "BingSearchMasterSearch"
            }
        )
        # Unprocessable Entity
        raise CustomException(
            f"Validation error in 'master_search': {str(e)}", error_code=422, action_field="BingSearchMasterSearch") from e

    except Exception as ex:
        logger.exception(
            "Unexpected error in 'master_search'", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "action_field": "BingSearchMasterSearch"
            }
        )
        # Internal Server Error
        raise CustomException(
            f"Unexpected error in 'master_search': {str(ex)}", error_code=500, action_field="BingSearchMasterSearch") from ex

# async wrapper for master_search


async def async_master_search(input_json):
    result = await master_search(input_json)
    return result

# Defining the functions
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
                        "description": "The keywords to search for in the specified operation.Maximum keyword should only be 50 "
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

# get_completion function with status code-based error handling

# Retry on APIConnectionError and RateLimitError


@backoff.on_exception(backoff.expo,
                      (openai.APIConnectionError, openai.RateLimitError),
                      # Max retry time in seconds (e.g., 5 minutes in seconds)
                      max_time=180,
                      logger=logger)
def make_openai_api_call(client, messages, settings, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, tools):
    # This function is wrapped with backoff to handle retry logic
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
    response = response.model_dump()
    return response


def get_completion(input_data, tools, temperature=0, max_tokens=4095, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):

    try:
        if isinstance(input_data, ChatHistory):
            messages = input_data.messages
        elif isinstance(input_data, list):
            messages = input_data
        else:
            raise ValueError(
                "Invalid Data Format. Expected ChatHistory or list of Messages object.")
    except ValidationError as e:
        logger.error(f"ChatHistory validation error.", exc_info=True,
                     extra={
                         "correlation_id": correlation_id,
                         "action_field": "GetCompletionCall",
                     }
                     )
        raise CustomException("Invalid ChatHistory data.",
                              error_code=400) from e
    except Exception as e:
        logger.error(f"ChatHistory Exception.", exc_info=True,
                     extra={

                         "correlation_id": correlation_id,
                         "action_field": "GetCompletionCall",
                     }
                     )
        raise CustomException("Invalid ChatHistory data.",
                              error_code=400) from e
    try:
        settings = Settings()
        client = AzureOpenAI(api_version=settings.api_version,
                             api_key=settings.api_key,
                             base_url=settings.base_url)
        logger.info("AzureOpenAI client defined.",
                    extra={

                        "correlation_id": correlation_id,
                        "action_field": "GetCompletionCall",
                    }
                    )

    except ValidationError as e:
        logger.error(
            f"OpenAI Configuration validation error.", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "action_field": "GetCompletionCall",
            }
        )
        raise CustomException(
            "OpenAI Configuration validation error.", error_code=500) from e

    validated_chat_history = ChatHistory(messages=messages)
    messages = validated_chat_history.model_dump()['messages'] or []

    logger.info("Sending completion request to OpenAI API.",
                extra={

                    "correlation_id": correlation_id,
                    "action_field": "GetCompletionCall",
                }
                )
    logger.debug(f"Messages: {messages}", exc_info=True,
                 extra={

                     "correlation_id": correlation_id,
                     "action_field": "GetCompletionCall",
                 }
                 )
    logger.debug(f"Functions: {tools}", exc_info=True,
                 extra={

                     "correlation_id": correlation_id,
                     "action_field": "GetCompletionCall",
                 }
                 )

    # Make API call with retry and backoff logic
    try:
        start_time = time.time()
        response = make_openai_api_call(
            client, messages, settings, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, tools=tools)
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        logger.info("Received response from OpenAI API.",
                    extra={
                        "correlation_id": correlation_id,
                        "action_field": "OpenAIAPICall",
                        "response_time": response_time_ms
                    }
                    )
        return response['choices'][0]['message'], response['usage']

    except CustomException as e:
        raise

    except openai.APIConnectionError as e:
        logger.exception(f"API Connection Error.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        # Service Unavailable
        raise CustomException(
            f"Failed to connect to OpenAI API. {str(e)}", error_code=503) from e

    except openai.AuthenticationError as e:
        logger.exception(f"Authentication Error.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "GetCompletionCall",
                         }
                         )
        raise CustomException(
            f"Authentication failed.{str(e)}", error_code=401) from e  # Unauthorized

    except openai.RateLimitError as e:
        logger.exception(f"Rate Limit Exceeded.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        raise CustomException(
            f"API rate limit exceeded. {str(e)}", error_code=429) from e  # Too Many Requests

    except openai.APITimeoutError as e:
        logger.exception(f"Request Timeout.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        raise CustomException(
            f"Request timed out. {str(e)}", error_code=504) from e  # Gateway Timeout

    except openai.BadRequestError as e:
        logger.exception(f"Bad Request.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        raise CustomException(
            f"Bad request, please check parameters. {str(e)}", error_code=400) from e  # Bad Request

    except openai.ConflictError as e:
        logger.exception(f"Conflict Error.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        raise CustomException(
            f"Resource conflict error. {str(e)}", error_code=409) from e  # Conflict

    except openai.InternalServerError as e:
        logger.exception(
            f"Internal Server Error.", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "action_field": "OpenAIAPICall",
            }
        )
        # Internal Server Error
        raise CustomException(
            f"OpenAI server error. {str(e)}", error_code=500) from e

    except openai.NotFoundError as e:
        logger.exception(f"Not Found.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        raise CustomException(
            f"Requested resource not found. {str(e)}", error_code=404) from e  # Not Found

    except openai.PermissionDeniedError as e:
        logger.exception(f"Permission Denied.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        raise CustomException(
            f"Permission denied. {str(e)}", error_code=403) from e  # Forbidden

    except Exception as e:
        logger.exception(f"Unexpected Error.", exc_info=True,
                         extra={

                             "correlation_id": correlation_id,
                             "action_field": "OpenAIAPICall",
                         }
                         )
        # Internal Server Error
        raise CustomException(
            f"Unexpected error occurred while using the OpenAI API. {str(e)}", error_code=500) from e

##########################################################################################################################


def get_answer(chat_history: json):
    message = chat_history.model_copy()
    try:
        response, usage = get_completion(
            input_data=message, tools=tools)
    except CustomException as e:
        raise e

    # Introduce a maximum number of iterations to prevent infinite loops
    MAX_LOOP_ITERATIONS = 15
    iteration_counter = 0

    while True:
        iteration_counter += 1
        if iteration_counter > MAX_LOOP_ITERATIONS:
            logger.error("Exceeded maximum iterations in tool call processing loop.", exc_info=True,
                         extra={
                             "correlation_id": correlation_id,
                             "action_field": "GetAnswerCall: Tool Call Iteration exceeded.",
                         })
            raise CustomException(
                "Exceeded maximum iterations in tool call processing loop.",
                500,
                action_field="FunctionToolCall"
            )

        # Assuming 'response' contains the assistant's output with tool_calls
        if response.get('tool_calls'):
            for tool_call in response['tool_calls']:
                tool_call_id = tool_call['id']
                tool_name = tool_call['function']['name']
                function_response = ""

                try:
                    # Execute tool call logic based on tool_name
                    if tool_name == "master_search":
                        arguments = json.loads(
                            tool_call['function']['arguments'])
                        function_response = asyncio.run(
                            async_master_search(json.dumps(arguments))
                        )

                    # Append the response message for the tool call
                    message.messages.append(
                        Message(
                            tool_call_id=tool_call_id,
                            role="assistant",
                            name=tool_name,
                            content=function_response
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing tool call {tool_call_id}: {e}")
                    raise

            # Regenerate assistant response
            next_response, usage = get_completion(
                input_data=message, tools=tools)
            return next_response['content'], usage

        else:
            # If no tool calls, finalize response
            message.messages.append(
                Message(
                    role="assistant",
                    content=response.get('content', ""),
                )
            )
            return response['content'], usage

##########################################################################################################################

@app.route(route="knowledgeAgent")
def knowledgeAgent(req: func.HttpRequest) -> func.HttpResponse:
    logger.info(
        'Python HTTP trigger function processed a request.',
        extra={

            "correlation_id": correlation_id,#for tracking insights with unique correlatoin_id.
            "action_field": "MainCall",
        }
    )

    try:
        chat_history_json = req.get_json()

    except ValueError as e:
        logger.error(f"Invalid request format.", exc_info=True,
                     extra={

                         "correlation_id": correlation_id,
                         "user_name": None,
                         "action_field": "MainCall",
                     }
                     )
        error_logging(
            user_name=None,
            error_message=str(e),
            error_status=400,
            user_question=None,
            error_description="ValueError",
            action_field="Invalid JSON format"
        )
        return func.HttpResponse(json.dumps({"ValueError": str(e)}), status_code=400)
    try:

        chat_history_dict = {"messages": chat_history_json['request_message']}
        user_question = chat_history_json['request_message'][-1]['content']
        user_Details = chat_history_json['user_details']
        user_name = user_Details['username']

        chat_history = ChatHistory(**chat_history_dict)

    except ValidationError as e:
        logger.error(f"Invalid chat history format.", exc_info=True,
                     extra={

                         "correlation_id": correlation_id,
                         "user_name": user_name,
                         "action_field": "MainCall",
                     }
                     )
        error_logging(
            user_name=user_name,
            error_message=str(e),
            error_status=400,
            user_question=user_question,
            error_description="ValidationError",
            action_field="Invalid Chat History Format"
        )
        return func.HttpResponse(json.dumps({"ValidationError": str(e)}), status_code=400)

    try:
        ans, usage = get_answer(chat_history)
        logger.info(
            "Returned the response from get_answer() function.",
            extra={

                "correlation_id": correlation_id,
                "user_name": user_name,
                "action_field": "MainCall",
            }
        )
        logger.debug(
            f"Response from get_answer() function: {ans}", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "user_name": user_name,
                "action_field": "MainCall",
            }
        )

        user_Details['answer'] = ans
        user_Details['question'] = user_question
        user_Details['usage'] = usage

        #save_document(user_Details)
        
        return func.HttpResponse(json.dumps({"role": "assistant", "content": ans, "usage": usage}), status_code=200)

    except CustomException as e:
        # Log stack trace and use the specific error code from CustomException
        logger.exception(
            "CustomException occurred in main while processing request.", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "user_name": user_name,
                "action_field": "GetAnswerCallCustomException",
            }
        )

        error_response = {
            "errorAt": datetime.utcnow().isoformat() + "Z",
            "role": "assistant",
            "error_description": str(e),
            "error_code": e.error_code,
            'action_field': e.action_field,
        }

        error_logging(
            user_name=user_name,
            error_message=str(e),
            error_status=e.error_code,
            user_question=user_question,
            error_description="CustomException",
            action_field=e.action_field
        )

        return func.HttpResponse(
            json.dumps(error_response),
            status_code=e.error_code  # Use the specific error code
        )

    except Exception as e:
        # Catch any other exceptions that weren't CustomException
        logger.exception(
            "Unexpected exception occurred in main.", exc_info=True,
            extra={

                "correlation_id": correlation_id,
                "user_name": user_name,
                "action_field": "GetAnswerCallUnknownException",
            }
        )

        error_response = {
            "role": "assistant",
            # "errorAt": datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "errorAt": datetime.utcnow().isoformat() + "Z",
            "error_description": str(e),
            "error_code": 500,
        }

        error_logging(
            user_name=user_name,
            error_message=str(e),
            error_status=500,
            user_question=user_question,
            error_description="UnexpectedError",
            action_field=None
        )

        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500
        )

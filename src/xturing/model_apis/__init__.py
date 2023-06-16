from xturing.model_apis.ai21 import AI21TextGenerationAPI
from xturing.model_apis.ai21 import J2Grande as AI21J2Grande
from xturing.model_apis.base import BaseApi, TextGenerationAPI
from xturing.model_apis.cohere import CohereTextGenerationAPI
from xturing.model_apis.cohere import Medium as CohereMedium
from xturing.model_apis.openai import ChatGPT as OpenAIChatGPT
from xturing.model_apis.openai import Davinci as OpenAIDavinci
from xturing.model_apis.openai import OpenAITextGenerationAPI

BaseApi.add_to_registry(OpenAITextGenerationAPI.config_name, OpenAITextGenerationAPI)
BaseApi.add_to_registry(CohereTextGenerationAPI.config_name, CohereTextGenerationAPI)
BaseApi.add_to_registry(AI21TextGenerationAPI.config_name, AI21TextGenerationAPI)
BaseApi.add_to_registry(OpenAIDavinci.config_name, OpenAIDavinci)
BaseApi.add_to_registry(OpenAIChatGPT.config_name, OpenAIChatGPT)
BaseApi.add_to_registry(CohereMedium.config_name, CohereMedium)
BaseApi.add_to_registry(AI21J2Grande.config_name, AI21J2Grande)

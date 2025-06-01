# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import openai
import dashscope
from openai import AzureOpenAI
import transformers
import time


class LMMBaseModel(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for predict method.
    """

    def __init__(self, model_name, max_new_tokens=8192, temperature=0, device='auto'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      temperature=self.temperature,
                                      do_sample=True,
                                      **kwargs)

        out = self.tokenizer.decode(outputs[0])
        return out

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)


class GLMModel(LMMBaseModel):
    """
    Language model class for the GLM model.

    Inherits from LMMBaseModel and sets up the Baichuan language model for use.

    Parameters:
    -----------
    model : str
        The name of the Baichuan model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(GLMModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/' + self.model_name, torch_dtype=dtype,
                                                       device_map=device, trust_remote_code=True)
        self.model = AutoModel.from_pretrained('/root/autodl-tmp/' + self.model_name, torch_dtype=dtype,
                                               device_map=device, trust_remote_code=True)

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=False,  # todo:for GLM,使用其他模型时改掉
                                      **kwargs)

        out = self.tokenizer.decode(outputs[0])
        return out.split("答案：", 1)[-1]


class BaichuanModel(LMMBaseModel):
    """
    Language model class for the Baichuan model.

    Inherits from LMMBaseModel and sets up the Baichuan language model for use.

    Parameters:
    -----------
    model : str
        The name of the Baichuan model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(BaichuanModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/' + self.model_name, torch_dtype=dtype,
                                                       device_map=device, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/' + self.model_name, torch_dtype=dtype,
                                                          device_map=device, trust_remote_code=True)

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=True,
                                      **kwargs)

        out = self.tokenizer.decode(outputs[0])
        return out.split("答案：", 1)[-1]


class ChineseLlama3Model(LMMBaseModel):
    """
    ChineseLlama3Model

    Inherits from LMMBaseModel and sets up the Llama language model for use.

    Parameters:
    -----------
    model : str
        The name of the Llama model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    system_prompt : str
        The system prompt to be used (default is None).
    model_dir : str
        The directory containing the model files (default is None). If not provided, it will be downloaded from the HuggingFace model hub.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(ChineseLlama3Model, self).__init__(model_name, max_new_tokens, temperature, device)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="/root/autodl-tmp/FlagAlpha/" + model_name,
            model_kwargs={"torch_dtype": torch.float16},
            device=device,
        )
        self.pipeline.model.config.pad_token_id = self.pipeline.model.config.eos_token_id

    def predict(self, input_text, **kwargs):
        messages = [{"role": "system", "content": ""}]
        messages.append(
            {"role": "user", "content": input_text}
        )

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.pipeline(
            prompt,
            max_new_tokens=8712,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0,
            top_p=0.9
        )

        content = outputs[0]["generated_text"][len(prompt):]

        return content


class FaruiModel(LMMBaseModel):
    """
    Language model class for interfacing with Farui models (e.g., tongyifarui-890, farui-plus).

    Parameters:
    -----------
    model_name : str
        The name of the Farui model to be used.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    system_prompt : str
        The system prompt to be used.

    Methods:
    --------
    predict(input_text, **kwargs)
        Sends a request to the Farui model and returns the response.
    """

    def __init__(self, model_name, max_new_tokens, temperature, system_prompt):
        super(FaruiModel, self).__init__(model_name, max_new_tokens, temperature)
        self.system_prompt = system_prompt

    def predict(self, input_text, **kwargs):
        # Prepare system prompt
        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}

        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]
        messages.insert(0, system_messages)

        try:
            # Call Dashscope API
            response = dashscope.Generation.call(
                self.model_name,  # Model name (e.g., "tongyifarui-890", "farui-plus")
                messages=messages,
                result_format='message',  # Set the result to "message" format
            )
            response = response["output"]["choices"][0].message.content
        except Exception as e:
            print(f"Error: {e}+{messages}")
            response = ""
        return response


@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(2))
def Azure_create_chat_completion_with_backoff(messages, client, deployment_name):
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0,
        )
    except Exception as e:
        return f"Error: {e}+{messages}"
    return response


class Azure_OpenAIModel(LMMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models.

    Inherits from LMMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """

    def __init__(self, model_name, max_new_tokens, temperature, system_prompt, openai_key, base_url, deployment_name):
        super(Azure_OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)
        self.base_url = base_url
        self.openai_key = openai_key
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt

    def predict(self, input_text, **kwargs):
        # client = openai.OpenAI(base_url=self.base_url, api_key=self.openai_key)
        openai_client = AzureOpenAI(
            azure_endpoint=self.base_url,
            api_key=self.openai_key,
            api_version="2023-05-15",
        )
        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}

        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]

        messages.insert(0, system_messages)

        deployment_name = self.deployment_name

        response = Azure_create_chat_completion_with_backoff(
            messages=messages,
            client=openai_client,
            deployment_name=deployment_name,
        )

        if "Error" in response or response == None:
            result = response
        else:
            result = response.choices[0].message.content

        return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def create_chat_completion_with_backoff(client, model, messages, temperature, max_tokens, n):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
    except Exception as e:
        print(f"Error: {e}+{messages}")
        response = ""
    return response


class OpenAIModel(LMMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models.

    Inherits from LMMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """

    def __init__(self, model_name, max_new_tokens, temperature, system_prompt, openai_key, base_url):
        super(OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)
        self.base_url = base_url
        self.openai_key = openai_key
        self.system_prompt = system_prompt

    def predict(self, input_text, **kwargs):
        client = openai.OpenAI(base_url=self.base_url, api_key=self.openai_key)

        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}

        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]

        messages.insert(0, system_messages)

        # extra parameterss
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens

        response = create_chat_completion_with_backoff(
            client=client,
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n,
        )
        if response == "":
            result = ""
        else:
            if n > 1:
                result = [choice.message.content for choice in response.choices]
            else:
                result = response.choices[0].message.content

        return result

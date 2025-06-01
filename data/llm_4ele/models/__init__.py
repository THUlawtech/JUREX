# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .models import *

# MAXLEN = {"ZhipuAI/chatglm3-6b": 8192, "Azure-gpt35": 16385,
#           "baichuan-inc/Baichuan2-7B-Chat": 4096,
#           "LawyerLLaMA": 2048, "Llama3-Chinese-8B-Instruct": 8192, "tongyifarui-890": 12000}

# A dictionary mapping of model architecture to its supported model names
MODEL_LIST = {
    GLMModel: ["ZhipuAI/chatglm3-6b"],
    ChineseLlama3Model: ["Llama3-Chinese-8B-Instruct"],
    OpenAIModel: ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-2024-05-13', 'gpt-4o-mini','gpt-3.5-turbo-1106','gpt-4o-2024-08-06','qwen-plus-0806','qwen-plus'],
    BaichuanModel: ['baichuan-inc/Baichuan2-7B-Base', 'baichuan-inc/Baichuan2-13B-Base',
                    'baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-13B-Chat'],
    FaruiModel: ['tongyifarui-890', 'farui-plus', 'qwen2.5-72b-instruct'],
    Azure_OpenAIModel: ['Azure-gpt35'],
}

SUPPORTED_MODELS = [model for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]]


class LLModel(object):
    """
    A class providing an interface for various language models.

    This class supports creating and interfacing with different language models, handling prompt engineering, and performing model inference.

    Parameters:
    -----------
    model : str
        The name of the model to be used.
    max_new_tokens : int, optional
        The maximum number of new tokens to be generated (default is 20).
    temperature : float, optional
        The temperature for text generation (default is 0).
    device : str, optional
        The device to be used for inference (default is "cuda").
    dtype : str, optional
        The loaded data type of the language model (default is "auto").
    model_dir : str or None, optional
        The directory containing the model files (default is None).
    system_prompt : str or None, optional
        The system prompt to be used (default is None).
    api_key : str or None, optional
        The API key for API-based models (GPT series and Gemini series), if required (default is None).

    Methods:
    --------
    _create_model(max_new_tokens, temperature, device, dtype, model_dir, system_prompt, api_key)
        Creates and returns the appropriate model instance.
    convert_text_to_prompt(text, role)
        Constructs a prompt based on the text and role.
    concat_prompts(prompt_list)
        Concatenates multiple prompts into a single prompt.
    _gpt_concat_prompts(prompt_list)
        Concatenates prompts for GPT models.
    _other_concat_prompts(prompt_list)
        Concatenates prompts for non-GPT models.
    __call__(input_text, **kwargs)
        Makes a prediction based on the input text using the loaded model.
    """

    @staticmethod
    def model_list():
        return SUPPORTED_MODELS

    def __init__(self, model, max_new_tokens=2000, temperature=0, device="cuda", dtype="auto",
                 system_prompt=None, api_key=None, base_url=None, deployment_name=None):
        self.model_name = model
        self.model = self._create_model(max_new_tokens, temperature, device, dtype, system_prompt, api_key=api_key,
                                        base_url=base_url,deployment_name=deployment_name)

    def _create_model(self, max_new_tokens, temperature, device, dtype, system_prompt, api_key, base_url, deployment_name=None):
        """Creates and returns the appropriate model based on the model name."""

        # Dictionary mapping of model names to their respective classes
        model_mapping = {model: model_class for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]}

        # Get the model class based on the model name and instantiate it
        model_class = model_mapping.get(self.model_name)
        if model_class:
            if model_class in [OpenAIModel]:
                return model_class(self.model_name, max_new_tokens, temperature, system_prompt, api_key, base_url)
            elif model_class in [Azure_OpenAIModel]:
                return model_class(self.model_name, max_new_tokens, temperature, system_prompt, api_key, base_url,
                                   deployment_name)
            else:
                return model_class(self.model_name, max_new_tokens, temperature, system_prompt)
        else:
            raise ValueError("The model is not supported!")

    def convert_text_to_prompt(self, text, role):
        """Constructs multi_turn conversation for complex methods in prompt engineering."""
        if self.model_name == ['gpt4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo',
                               'Llama3-Chinese-8B-Instruct']:
            return {'role': role, 'content': text}
        else:
            # return str(role) + ': ' + str(text) + '\n'
            return str(text) + '\n'

    def concat_prompts(self, prompt_list):
        """Concatenates the prompts into a single prompt."""
        if self.model_name == ['gpt4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo']:
            return self._gpt_concat_prompts(prompt_list)
        else:
            return self._other_concat_prompts(prompt_list)

    def _gpt_concat_prompts(self, prompt_list):
        """
        Concatenate prompts from various inputs into a single list of dictionaries.

        The function accepts any number of keyword arguments, each of which can be either
        a dictionary or a list of dictionaries. It concatenates all inputs into a single list.

        Returns:
            A list of dictionaries containing all the prompts from the input arguments.
        """
        # Initialize an empty list to hold all dictionaries
        all_prompts = []

        # Iterate over each keyword argument
        for arg in prompt_list:
            # Check if the argument is a dictionary, and if so, add it to the list
            if isinstance(arg, dict):
                all_prompts.append(arg)
            # Check if the argument is a list of dictionaries
            elif isinstance(arg, list) and all(isinstance(item, dict) for item in arg):
                # Extend the list with the dictionaries from the current argument
                all_prompts.extend(arg)
            else:
                raise ValueError("All arguments must be dictionaries or lists of dictionaries.")

        return all_prompts

    def _other_concat_prompts(self, prompt_list):
        """
        Concatenate prompts from various inputs into a single strings.

        The function accepts any number of keyword arguments, each of which must be
        a string. It concatenates all inputs into a single string.

        Returns:
            A string containing all the prompts from the input arguments.
        """
        # Initialize an empty string to hold all prompts
        all_prompts = ""

        # Iterate over each keyword argument
        for arg in prompt_list:
            # Check if the argument is a string, and if so, add it to the list
            if isinstance(arg, str):
                all_prompts = all_prompts + '\n' + arg
            else:
                raise ValueError("All arguments must be strings.")

        return all_prompts

    def __call__(self, input_text, **kwargs):
        """Predicts the output based on the given input text using the loaded model."""
        return self.model.predict(input_text, **kwargs)

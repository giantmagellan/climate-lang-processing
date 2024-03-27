import os
import pandas as pd

from openai import OpenAI


def get_llm_response(prompt: str, model_url: str) -> str:
    """
    Get response from ChatGPT.
    :param prompt: str, instruction prompt for LLM
    :param model_url: str, model name
    :return: str, LLM response
    """
    
    if model_url in ["gpt-3.5-turbo-0125", "gpt-4"]:
        client = OpenAI(api_key=os.environ.get('OPEN_AI_KEY'))

        response = client.chat.completions.create(
            model = model_url,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.7,
        )
        return response.choices[0].message.content 
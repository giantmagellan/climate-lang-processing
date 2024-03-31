import time
import os
import pandas as pd
from openai import OpenAI

from helper import build_prompt


def generate_llm_output(payload: dict=None) -> str:
    """
    Generates response from LLM.
    :param payload: dict, 
    """
    start_time = time.time()
    try:
        if not snippet:
            return "No news snippet to analyze."
        else:
            # Create LLM prompt
            snippet = payload["snippet"]
            sys_prompt, intro, instruct = payload['system_prompt'], payload['introduction'], payload['instruction']
            prompt = build_prompt(sys_prompt, intro, instruct, snippet)

            model_urls = [
                "gpt-3.5-turbo-0125", 
                "gpt-4"
                ]
            return get_llm_response(prompt, model_urls[0])
        
    except Exception as e:
        print(f"An error {e} occurred when generating response.")
    
    print(f"Elapsed runtime: {round(time.time() - start_time, 2)} seconds.")


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
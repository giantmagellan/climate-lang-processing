import time
import os
import json
import pandas as pd


class PromptConstructor:
    """
    A utility class for constructing prompts dynamically from a JSON file.
    """
        
    @staticmethod
    def _load_prompts(filename: str='prompts.json', directory: str=None) -> dict:
        """
        Load prompt from local JSON file.
        :param file: str, name of JSON file.
        :return: dict, prompts
        """
        CURRENT_DIRECTORY = directory or os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(CURRENT_DIRECTORY, filename)

        try: 
            with open(file_path, 'r') as f: return json.load(f)
        except FileNotFoundError: raise FileNotFoundError("Prompt file not found.")
        except json.JSONDecodeError as e: raise ValueError("Error decoding JSON in file %s", e)

    @classmethod
    def construct_prompt(cls, filename: str=None) -> str:
        """
        Build final input prompt.
        :param prompt: str, system and user prompts
        :param filename: str, name of JSON file 
        :return: str, combined prompts for LLM request 
        """
        prompts = cls._load_prompts(filename)

        try:
            # Converting role headers to python to utilize formatting 
            HEADER_ID = prompts["HEADER_ID"]
            HEADER_ID_SYS = eval(prompts["HEADER_ID_SYS"])
            HEADER_ID_USER = eval(prompts["HEADER_ID_USER"])
            HEADER_ID_ASSIST = eval(prompts["HEADER_ID_ASSIST"])
            B_TEXT = prompts["B_TEXT"]
            EOT_ID = prompts["EOT_ID"]
        
            # Assign special tokens to system and user prompt
            sys_prompt = prompts["system"].format_map({
                'B_TEXT': B_TEXT,
                'HEADER_ID_SYS': HEADER_ID_SYS,
                'EOT_ID': EOT_ID
            })
            user_prompt = prompts["user"].format_map({
                'HEADER_ID_USER': HEADER_ID_USER,
                'EOT_ID': EOT_ID
            })

            return f"{sys_prompt}{user_prompt}\n\n{HEADER_ID_ASSIST}"
        
        except KeyError as e: raise KeyError("Missing special tokens %s", e)
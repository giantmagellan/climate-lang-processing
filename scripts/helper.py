import os
import pandas as pd

from prompts import B_SYS, E_SYS, B_INST, E_INST 
from prompts import DEFAULT_SYSTEM_PROMPT


def build_prompt(introduction: str, instructions: str, system_prompt: str=DEFAULT_SYSTEM_PROMPT, 
                 snippet: str=None) -> str:
    """
    Creates a prompt template by combining a default system prompt and
    a task-specific set of instructions.
    :param instruction: str, instructions for the model to perform.
    :param system_prompt: str, system prompt w/ ethical standards.
    :return transcript: str, transcript to be summarized.
    """
    try:
        system_prompt = f"{B_SYS}{system_prompt}{E_SYS}"
        prompt_template = f"{system_prompt}\n{introduction}"
        instructions = f"{B_INST}{instructions}{E_INST}"

        prompt = "".join([
            prompt_template,
            snippet, 
            instructions
        ])
        
        return prompt
    
    except ValueError:
        print("Improper inputs provided.")
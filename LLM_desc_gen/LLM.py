import os

from openai import OpenAI
from rich import print as rprint
import dashscope
from utility.utils import *
from http import HTTPStatus
import time
import openai
from google import genai

class chatLLM:
    def __init__(self, prompt_system, temperature=0.7) -> None:

        self.prompt_system = prompt_system
        self.temperature = temperature

    def chat(self, prompt_user, model="gpt-4o-2024-08-06", save_path=None, delay_seconds=0):
        
        self.prompt_user = prompt_user
        time.sleep(delay_seconds)
        
        if 'gemini' in model:
            client = genai.Client(api_key="key")

            response = client.models.generate_content(
                model=model, 
                contents=self.prompt_user
            )
            if save_path is not None:
                save_lines(save_path, [response.text])
                print(f"Successfully query {model} and save to {save_path}")
            return response.text
        else:
            if 'gpt' in model:
                os.environ["OPENAI_API_KEY"] = "key"
                os.environ["OPENAI_BASE_URL"] = "url"
                
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL")
            )
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.prompt_system},
                        {"role": "user", "content": self.prompt_user}
                    ]
                )
            
                answer = completion.choices[0].message.content
                if save_path is not None:
                    save_lines(save_path, [answer])
                    rprint(f"Successfully query {model} and save to {save_path}")
                return answer
            
            except openai.APITimeoutError as e:

                rprint(f"[bold red]Request to {model} timed out! Error message: {e}[/bold red]")

                return f"Request to model {model} timed out, no answer was obtained"



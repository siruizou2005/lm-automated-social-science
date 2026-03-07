from openai import OpenAI, RateLimitError
try:
    import replicate
except ImportError:
    replicate = None
from dotenv import load_dotenv
import os
import re
import sys
import json
import random
from json.decoder import JSONDecodeError
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Dict, List, Union

load_dotenv()

sys.path.append('./src/Serialization')
# current_script_path = os.path.dirname(os.path.abspath(__file__))

from Serialize import RegisteredSerializable

class LanguageModel(RegisteredSerializable):
    def __init__(self, model: str, family: str, temperature: float, max_tokens = None, system_prompt: str = "") -> None:
        self.model: str = model
        self.family: str = family
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.system_prompt: str = system_prompt
        self.client = None

        # openrouter accepts any model name; all other families require explicit registration
        self.family_model_mapping = {
            "openai": {
                "text-davinci-003": 'call_openai_api',
                "gpt-3.5-turbo": 'call_openai_api_35',
                "gpt-4": 'call_openai_api_35',
            },
            "replicate": {
                "llama70b-v2-chat": 'call_llama70b_v2',
                "llama13b-v2-chat": 'call_llama13b_v2',
            },
        }

        if self.family == "openrouter":
            self.call_llm = self.call_openai_api_35
        elif self.family in self.family_model_mapping:
            if self.model not in self.family_model_mapping[self.family]:
                raise ValueError(f"Model '{model}' not supported for the '{family}' family.")
            self.call_llm = getattr(self, self.family_model_mapping[self.family][self.model])
        else:
            raise ValueError(f"Family '{family}' not supported.")

    def _get_openai_client(self) -> OpenAI:
        if self.client is None:
            # base_url defaults to OpenAI; set OPENAI_BASE_URL in .env to use OpenRouter or any compatible endpoint
            # timeout: 120s per request (default is 600s); max_retries=0 because tenacity handles retries
            client_kwargs = {
                "api_key": os.getenv('OPENAI_API_KEY'),
                "timeout": float(os.getenv('OPENAI_TIMEOUT', '120')),
                "max_retries": 0,
            }
            if base_url := os.getenv('OPENAI_BASE_URL'):
                client_kwargs["base_url"] = base_url
            else:
                organization = os.getenv('ORGANIZATION_ID')
                if organization:
                    client_kwargs["organization"] = organization
            self.client = OpenAI(**client_kwargs)
        return self.client

    def _get_replicate_module(self):
        if replicate is None:
            raise ImportError(
                "replicate package is required to use the replicate model family."
            )
        return replicate


    def __repr__(self):
        string = f'''Family: {self.family}\nModel: {self.model}\nTemperature: {self.temperature}'''
        return string

    def list_valid_LLMs(self) -> None:
        for family, model in self.family_model_mapping.items():
            print(f'{family}: {list(model.keys())}')

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(100))
    def call_openai_api_35(self, prompt: str, response_format: dict = None) -> str:
        try:
            kwargs = dict(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt},
                           {"role": "user",   "content": prompt}],
                temperature=self.temperature,
            )
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = self._get_openai_client().chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except RateLimitError as e:
            logging.warning("Rate limit exceeded. Retrying...")
            raise
        except Exception as e:
            logging.warning("API call failed (%s: %s). Retrying...", type(e).__name__, e)
            raise

    def call_llm_json(self, prompt: str) -> str:
        """Call the LLM with JSON mode enabled; guarantees a valid JSON string back."""
        return self.call_openai_api_35(prompt, response_format={"type": "json_object"})

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(100))
    def call_openai_api(self, prompt: str) -> str:
        # text-davinci-003 (Completion API) is discontinued; route through chat completions
        try:
            response = self._get_openai_client().chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            logging.warning("Rate limit exceeded. Retrying...")
            raise
        except Exception as e:
            logging.warning("API call failed (%s: %s). Retrying...", type(e).__name__, e)
            raise

    def call_llama70b_v2(self, prompt: str) -> str:
        replicate_module = self._get_replicate_module()
        model = "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"
        output = replicate_module.run(model,
                        input={"prompt":prompt,
                        "top_p": 1,
                        "system_prompt": """You are a helpful assistant.""",
                        "temperature": self.temperature, 
                        "max_length": 500,
                        "repetition_penalty": 1.5
                        }
                    )
        result = ''.join(output)
        return result
    
    def call_llama13b_v2(self, prompt: str, top_p: float = 1, max_length: int = 500, repetition_penalty: float = 1) -> str:
        replicate_module = self._get_replicate_module()
        model = "a16z-infra/llama-2-13b-chat:d5da4236b006f967ceb7da037be9cfc3924b20d21fed88e1e94f19d56e2d3111"
        output = replicate_module.run(model,
                        input={"prompt":prompt,
                        "top_p": 1,
                        "system_prompt": """You are a helpful assistant.""",
                        "temperature": self.temperature, 
                        "max_length": 500,
                        "repetition_penalty": 1.5
                        }
                    )
        result = ''.join(output)
        return result
        
class LLMMixin:
    def add_LLM(self, LLM: 'LanguageModel') -> None:
        self.LLM = LLM

    def call_llm(self, prompt: str) -> str:
        if hasattr(self, 'LLM'):
            return self.LLM.call_llm(prompt)
        else:
            raise NotImplementedError("This method gets implemented when you add an LLM")

    def call_llm_json(self, prompt: str) -> str:
        """Call LLM in JSON mode; returns a guaranteed valid JSON string."""
        if hasattr(self, 'LLM'):
            return self.LLM.call_llm_json(prompt)
        else:
            raise NotImplementedError("This method gets implemented when you add an LLM")
        
def _strip_markdown_json(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text


def _normalize_json_text(text: str) -> str:
    """Normalize smart quotes so JSON emitted by chat models remains parseable."""
    return text.translate(
        str.maketrans(
            {
                "\u201c": '"',
                "\u201d": '"',
                "\u2018": "'",
                "\u2019": "'",
            }
        )
    )


def llm_json_loader(raw_llm_output: str) -> Dict[str, str]:
    """Parse JSON from LLM output, stripping markdown fences and retrying with corrector on failure."""
    # Strip markdown fences before first attempt (handles models that wrap JSON in ```json blocks)
    llm_output: str = _normalize_json_text(_strip_markdown_json(raw_llm_output))
    for attempt in range(3):
        try:
            llm_json = json.loads(llm_output.lower())
            return llm_json

        except JSONDecodeError as e:
            print(f'Attempt {attempt+1}: invalid JSON, calling corrector. Output: {llm_output}', e.doc, e.pos)
            error_doc, error_pos = e.doc, e.pos

        try:
            llm_output = _strip_markdown_json(json_corrector(llm_output, error_doc, error_pos))
        except Exception as e:
            print(f'Attempt {attempt+1}/3: corrector error: {str(e)}')

    print(f'ORIGINAL STRING FROM LLM: {raw_llm_output}')
    print(f'FINAL FAILED STRING FROM LLM: {llm_output}')
    raise JSONDecodeError("INVALID JSON", llm_output, 0)

def make_llm(role: str, temperature: float, system_prompt: str = "") -> "LanguageModel":
    """Create a LanguageModel using role-based env vars.

    Roles: 'scientist' (SCM/agent design) or 'subject' (simulation agents).
    Env vars: LLM_{ROLE}_FAMILY and LLM_{ROLE}_MODEL (case-insensitive role).
    Falls back to openai/gpt-4 if not set.
    """
    key = role.upper()
    family = os.getenv(f"LLM_{key}_FAMILY", "openai")
    model  = os.getenv(f"LLM_{key}_MODEL",  "gpt-4")
    return LanguageModel(family=family, model=model, temperature=temperature, system_prompt=system_prompt)


def json_corrector(llm_output: str, error_doc, error_pos) -> str:
    """Send invalid JSON to LLM (JSON mode) for correction and return the fixed string."""
    LLM = make_llm("scientist", temperature=0.1)
    cleanup_prompt = (
        f"The following JSON is invalid: {llm_output}\n"
        f"Error: {error_doc} at position {error_pos}.\n"
        "Fix all errors while preserving the original information. "
        "Return only the corrected JSON with no extra text."
    )
    llm_cleaned_output = LLM.call_llm(cleanup_prompt)
    return _normalize_json_text(_strip_markdown_json(llm_cleaned_output)).lower()



if __name__ == "__main__":
    if False:
        """Test that the LLM can be serialized and deserialized"""
        params = {
            "family":"openai",
            "model": "text-davinci-003", #"gpt-3.5-turbo", 
            "temperature": 0.7   
        }
        LLM = LanguageModel(**params)
        json_str = LLM.serialize()
    if False:
        LLM = LanguageModel(family = 'replicate', model = 'llama13b-v2-chat', temperature = .1)
        output = LLM.call_llm('''In the following scenario: "A person getting cognitive behavioral therapy",
    Who are the individual human agents in a simple simulation of this scenario?
    The agents should have specified roles.
    For example, if the scenario was "negotiating to buy a car", then the agents should not be ["person 1", "person 2"], but should be ["seller", "buyer"]
    Only include agents that would speak during the scenario.
    Respond with a list of individual human agents, with the roles as their titles.
    Do not include a plurality of agents as a single agent in the list.
    Evey item in the list must be a singular agent, even if this makes the list long.
    For example, if the scenario was "a criminal case" the correct list of roles would be:
    ["judge", "defendant", "prosecutor", "defense attorney", "juror 1", "juror 2"]
    And the following would be incorrect: 
    ["judge", "defendant", "lawyers", "jurers"]
    You should respond with a python list as displayed in the correct example.
    Respond with a JSON in the following format and do not include any other text outside of the json:
    {"agents": ["agent 1", "agent 2", "agent 3", "agent 4"],
    "explanation": "explanation for choice of agents"}''')
        print(output)
        # print(LLM.call_llm('hey how are you?'))
        LLM.list_valid_LLMs()
        
    if True:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": ""},
                      {"role": "user", "content": "hey how are you?"}],
            max_tokens=256,
            temperature=0.5,
        )
        print(response)

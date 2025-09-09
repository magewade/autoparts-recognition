from dataclasses import asdict, is_dataclass
import re
import google.generativeai as genai
import logging
import time

# Universal prompt for description model
DESCRIPTION_MODEL_PROMPT = (
    "Extract the car brand (not a specific model or modification) and all part numbers from the following description. "
    "If the model name is written in a language other than English (for example, in Chinese), always translate or adapt it to the most common English name for that car brand. "
    "If the description mentions a specific model (like Rio, Golf, Camry, etc.), output only the general brand (like Kia, Volkswagen, Toyota, etc.) in the first field. "
    "Extract only numbers that look like real serial part numbers: they are usually 9 to 15 characters long, contain both letters and digits, and cannot be short (for example, three-four-digit numbers are not valid). Ignore numbers that are clearly too short or do not match this pattern. "
    "If there are several part numbers in the format like 03C906057DK/BH/AR (with slashes, commas, spaces, etc.), extract the first 5 of them, separated by commas. "
    "If there are more than 5 part numbers, output only the first five, then write 'etc' after them. "
    "IMPORTANT - If you extract more than one unique part number, this is a clear sign that the last field should be 'many'. "
    "If you extract only one part number, the last field should be 'one'. "
    "If you cannot extract any part number, write None in the second field. "
    "Output strictly in the format: Brand | part_number1, part_number2, ... | one/many. "
    "If you don't know, output: unknown | None | one. "
    "Do not explain your answer. Always answer in English. "
)


# --- Строгая постобработка результата LLM
def clean_llm_output(guess):
    # Убираем скобки и лишние символы
    guess = guess.replace("[", "").replace("]", "")
    parts = [p.strip() for p in guess.split("|")]
    if len(parts) != 3:
        return "unknown | None | one"
    # Приводим model к unknown если any или none
    model = parts[0].lower()
    if model in ("any", "none", "unknown", ""):
        model = "unknown"
    # Парсим номера
    numbers = [
        n.strip()
        for n in re.split(r",|/|\\|\s", parts[1])
        if n.strip() and n.strip().lower() != "none"
    ]
    numbers_str = ", ".join(numbers) if numbers else "None"
    # Если номеров больше одного, всегда many
    one_or_many = parts[2].strip().lower()
    if len(numbers) > 1:
        one_or_many = "many"
    elif one_or_many != "many":
        one_or_many = "one"
    return f"{model} | {numbers_str} | {one_or_many}"


def usage_to_dict(usage):
    if usage is None:
        return {
            "prompt_token_count": None,
            "candidates_token_count": None,
            "total_token_count": None,
        }
    if isinstance(usage, dict):
        return usage
    if is_dataclass(usage):
        return asdict(usage)
    if hasattr(usage, "__dict__"):
        return vars(usage)
    try:
        return {
            k: getattr(usage, k)
            for k in dir(usage)
            if not k.startswith("_") and not callable(getattr(usage, k))
        }
    except Exception:
        return {
            "prompt_token_count": None,
            "candidates_token_count": None,
            "total_token_count": None,
        }


class GeminiDescriptionInference:
    def __init__(self, api_keys, model_name="gemini-2.5-flash-lite"):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = model_name
        self.last_successful_key_index = 0
        self.configure_api()
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 1,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 512,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
            ],
        )

    def configure_api(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 6000,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
            ],
            system_instruction=DESCRIPTION_MODEL_PROMPT,
        )

    def switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_api()

    # logging removed: switched to API key index

    def __call__(self, desc, return_usage=False):
        prompt = DESCRIPTION_MODEL_PROMPT + f"Description: {desc}"
        num_keys = len(self.api_keys)
        max_retries = 5
        # last_successful_key_index is now initialized in __init__
        for offset in range(num_keys):
            key_attempt = (self.last_successful_key_index + offset) % num_keys
            self.current_key_index = key_attempt
            self.configure_api()
            # logging removed: switched to API key index
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    guess = response.text.strip()
                    usage = None
                    if hasattr(response, "result") and hasattr(
                        response.result, "usage_metadata"
                    ):
                        usage = usage_to_dict(response.result.usage_metadata)
                    elif hasattr(response, "usage_metadata"):
                        usage = usage_to_dict(response.usage_metadata)
                    else:
                        usage = usage_to_dict(None)
                    logging.info(f"[LLM desc] Answer: {guess}")
                    time.sleep(2.1)
                    # If we get a valid answer, remember this key for next time
                    self.last_successful_key_index = key_attempt
                    if return_usage:
                        return clean_llm_output(guess), usage
                    return clean_llm_output(guess)
                except Exception as e:
                    if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                        logging.warning(
                            f"[LLM desc] Quota or rate limit error for API key {self.current_key_index}: {e}"
                        )
                        time.sleep(2.1)
                        continue
                    else:
                        logging.warning(f"[Desc LLM] Error: {e}")
                        time.sleep(2.1)
            logging.info(
                f"[LLM desc] Switching to next API key (index {key_attempt+1})"
            )
        if return_usage:
            return "ERROR", {
                "prompt_token_count": None,
                "candidates_token_count": None,
                "total_token_count": None,
            }
        return "ERROR"

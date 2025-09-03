import re


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


import google.generativeai as genai
import logging
import time


class GeminiDescriptionInference:
    def __init__(self, api_keys, model_name="gemini-2.5-flash-lite"):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = model_name
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

    def switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_api()
        logging.info(f"[Desc LLM] Switched to API key index: {self.current_key_index}")

    def __call__(self, desc):
        prompt = (
            "Extract the car model (any, not just from a fixed list) and all part numbers from the following description. "
            "If there are several part numbers in the format like 03C906057DK/BH/AR (with slashes, commas, spaces, etc.), extract all of them, separated by commas. "
            "Carefully read the text and, if there are any indirect signs that the seller is offering more than one physical item (e.g. words like 'set', 'kit', 'several', '2 pcs', 'for different models', 'multiple', 'набор', 'комплект', 'несколько', '2 шт', etc.), set the last field to 'many'. "
            "If you are not sure, set it to 'one'. "
            "If you cannot find a model or number, write 'None'. "
            "Output strictly in this format: model | numbers | one_or_many. "
            "Do not use any tags, brackets, or extra formatting. Only output the three fields separated by |. "
            f"Description: {desc}"
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                guess = response.text.strip()
                logging.info(f"[LLM desc] Ответ: {guess}")
                time.sleep(2.1)  # <= 30 запросов в минуту
                return clean_llm_output(guess)
            except Exception as e:
                if "quota" in str(e).lower():
                    self.switch_api_key()
                    time.sleep(2.1)
                else:
                    logging.warning(f"[Desc LLM] Error: {e}")
                    time.sleep(2.1)
        return "ERROR"

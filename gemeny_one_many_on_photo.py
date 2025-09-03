import google.generativeai as genai
import logging
import time
import requests
from PIL import Image
from io import BytesIO


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
            "Extract the car model and part number(s) from the description. "
            "Always answer in English. Output strictly in this format: [model] | [number] | [one_or_many]. "
            "The model should be from this list: audi, toyota, nissan, suzuki, honda, daihatsu, subaru, mazda, bmw, lexus, volkswagen, volvo, mini, fiat, citroen, renault, ford, isuzu, opel, mitsubishi, mercedes, jaguar, peugeot, porsche, alfa_romeo, chevrolet, denso, hitachi. "
            "If no model is specified, output exactly: unknown | None | one. "
            "[one_or_many] must be 'many' ONLY if several unique part numbers are listed (if it is absolutely clear from the description that several different physical parts are sold), otherwise 'one'. "
            "Do not use any tags, brackets, or special formatting. Only output the three fields separated by |. "
            f" Description: {desc}"
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                guess = response.text.strip()
                logging.info(f"[LLM desc] Ответ: {guess}")
                time.sleep(2.1)  # <= 30 запросов в минуту
                # Приводим к строгому формату: model | number | one_or_many
                parts = [p.strip() for p in guess.split("|")]
                # Если не три поля, fallback
                if len(parts) != 3:
                    return "unknown | None | one"
                # one_or_many строго one/many
                parts[2] = "many" if parts[2].lower() == "many" else "one"
                return " | ".join(parts)
            except Exception as e:
                if "quota" in str(e).lower():
                    self.switch_api_key()
                    time.sleep(2.1)
                else:
                    logging.warning(f"[Desc LLM] Error: {e}")
                    time.sleep(2.1)
        return "ERROR"


class GeminiPhotoOneManyInference:
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
                "max_output_tokens": 128,
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
        logging.info(f"[Photo LLM] Switched to API key index: {self.current_key_index}")

    def __call__(self, image_url):
        prompt = (
            "Look at the image and answer strictly 'one' if there is only one unique physical car part on the photo, "
            "or 'many' if there are several different physical car parts visible. "
            "Do not explain your answer, do not use any tags or extra text. Only output 'one' or 'many'."
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Скачиваем картинку
                response_img = requests.get(image_url, timeout=15)
                img = Image.open(BytesIO(response_img.content)).convert("RGB")
                # Отправляем картинку и промпт в LLM
                gemini_response = self.model.generate_content([prompt, img])
                answer = gemini_response.text.strip().lower()
                logging.info(f"[LLM photo one/many] Ответ: {answer}")
                time.sleep(2.1)
                if answer in ("one", "many"):
                    return answer
                # Если невалидно — fallback
                return "one"
            except Exception as e:
                if "quota" in str(e).lower():
                    self.switch_api_key()
                    time.sleep(2.1)
                else:
                    logging.warning(f"[Photo LLM] Error: {e}")
                    time.sleep(2.1)
        return "ERROR"

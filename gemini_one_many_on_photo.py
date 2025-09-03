import google.generativeai as genai
import logging
import time
import requests
import random
import io
from pathlib import Path


PHOTO_ONE_MANY_PROMPT = (
    "If there is more than one unique physical car part visible, output 'many'. "
    "If there is only one unique physical car part, output 'one'. "
    "Output only 'one' or 'many'. Do not explain your answer."
    "If you don't know the answer, output 'unknown'."
)


class GeminiPhotoOneManyInference:
    def __init__(self, api_keys, model_name="gemini-2.5-flash-lite"):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = model_name
        self.configure_api()
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 16,
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
            system_instruction=PHOTO_ONE_MANY_PROMPT,
        )

    def configure_api(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_api()
        logging.info(f"[Photo LLM] Switched to API key index: {self.current_key_index}")


    def get_response(self, img_data, retry=False):  
        max_retries = 10
        base_delay = 5
        for attempt in range(max_retries):
            try:
                if isinstance(img_data, io.BytesIO):
                    image_bytes = img_data.getvalue()
                else:
                    image_bytes = img_data.read_bytes()

                image_part = {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_bytes,
                    }
                }

                time.sleep(random.uniform(1, 2))
                response = self.model.generate_content([image_part])

                logging.info(f"[Photo LLM] Full response: {response}")

                # пробуем достать текст
                answer_text = None
                if response.candidates:
                    parts = response.candidates[0].content.parts
                    if parts:
                        answer_text = parts[0].text

                if not answer_text:
                    logging.warning("[Photo LLM] Empty answer from model")
                    return None

                return answer_text.strip().lower()

            except Exception as e:
                if "quota" in str(e).lower():
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    if delay > 300:
                        self.switch_api_key()
                        delay = base_delay
                    logging.warning(
                        f"[Photo LLM] Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logging.error(f"[Photo LLM] Error in get_response: {str(e)}")
                    raise

        logging.error("[Photo LLM] Max retries reached. Unable to get a response.")
        raise Exception("Max retries reached. Unable to get a response.")

    def __call__(self, image_path):
        self.configure_api()
        if image_path.startswith("http"):
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            img_data = io.BytesIO(response.content)
        else:
            img = Path(image_path)
            if not img.exists():
                raise FileNotFoundError(f"Could not find image: {img}")
            img_data = img

        max_attempts = 2
        for attempt in range(max_attempts):
            answer = self.get_response(img_data, retry=(attempt > 0))
            if not answer:
                logging.info("[Photo LLM] Empty answer, retrying...")
                continue
            logging.info(f"[LLM photo one/many] Ответ: {answer}")
            if answer in ("one", "many"):
                return answer
            logging.info(f"[Photo LLM] Invalid answer '{answer}', retrying...")

        logging.warning("[Photo LLM] All attempts failed or only invalid answer found.")
        return "one"  # fallback

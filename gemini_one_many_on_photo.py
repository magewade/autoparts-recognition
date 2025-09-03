import google.generativeai as genai
import logging
import time
import requests
import random
import io

PHOTO_ONE_MANY_PROMPT = (
    "You are an expert in analyzing car part images. "
    "Look at the image and answer strictly 'one' if there is only one unique physical car part on the photo, "
    "or 'many' if there are several different physical car parts visible. "
    "Do not explain your answer, do not use any tags or extra text. Only output 'one' or 'many'."
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
            system_instruction=PHOTO_ONE_MANY_PROMPT,
        )
        self.message_history = []

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
                image_parts = [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": (
                                img_data.getvalue()
                                if isinstance(img_data, io.BytesIO)
                                else img_data.read_bytes()
                            ),
                        }
                    },
                ]
                prompt_parts = []
                if retry:
                    prompt_parts.append(
                        "Previous answer did not match required format. STRICTLY output only 'one' or 'many'."
                    )
                full_prompt = image_parts + prompt_parts
                time.sleep(random.uniform(1, 2))
                chat = self.model.start_chat(history=self.message_history)
                response = chat.send_message(full_prompt)
                logging.info(
                    f"[Photo LLM] Main model response: {getattr(response, 'text', None)}"
                )
                self.message_history.append({"role": "user", "parts": full_prompt})
                self.message_history.append(
                    {"role": "model", "parts": [getattr(response, "text", None)]}
                )
                return getattr(response, "text", None)
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
            img_data = io.BytesIO(response.content)
        else:
            from pathlib import Path

            img = Path(image_path)
            if not img.exists():
                raise FileNotFoundError(f"Could not find image: {img}")
            img_data = img
        self.message_history = []
        max_attempts = 2
        for attempt in range(max_attempts):
            if attempt == 1:
                # Усиливаем prompt для строгого формата
                orig_prompt = self.model.system_instruction
                self.model.system_instruction = (
                    "Previous answer did not match required format. STRICTLY output only 'one' or 'many'.\n\n"
                    + orig_prompt
                )
            answer = self.get_response(img_data, retry=(attempt > 0))
            if not answer:
                logging.info(f"[Photo LLM] Empty answer, retrying...")
                if attempt == max_attempts - 1:
                    break
                continue
            answer = answer.strip().lower()
            logging.info(f"[LLM photo one/many] Ответ: {answer}")
            if answer in ("one", "many"):
                self.message_history = []
                return answer
            if attempt < max_attempts - 1:
                logging.info(f"[Photo LLM] Invalid answer '{answer}', retrying...")
        logging.warning("[Photo LLM] All attempts failed or only invalid answer found.")
        self.message_history = []
        return "one"

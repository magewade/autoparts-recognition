from dataclasses import asdict, is_dataclass
import google.generativeai as genai
import logging
import time
import requests
import random
import io
from pathlib import Path


# Новый промпт для одновременного определения one/many и наличия наклейки с баркодом и брендом
PHOTO_ONE_MANY_BARCODE_PROMPT = (
    "If there is only one unique physical car part, output 'one'. "
    "If there are few unique physical car details (more than one) visible, output 'many'. "
    "If there is a visible sticker or label with both a barcode and a brand name, output 'True', otherwise output 'False'. "
    "Output strictly in the format: one|True, one|False, many|True, or many|False. Do not explain your answer. If you don't know, output 'unknown|unknown'."
)


# Универсальная функция для приведения usage к dict
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


# --- Основная функция для пакетной обработки списка картинок ---
def process_images_one_many_and_barcode_label(
    image_paths, api_keys, model_name="gemini-2.0-flash-lite"
):
    """
    Для списка image_paths:
    - Для каждой картинки вызывает LLM с новым промптом.
    - Если хотя бы на одной many — прерывает обработку и возвращает накопленный список (до many включительно).
    - Возвращает список строк вида one|True, many|False и т.д.
    """
    model = GeminiPhotoOneManyBarcodeInference(api_keys, model_name=model_name)
    predictions = []
    token_stats = []
    for img_path in image_paths:
        try:
            pred, usage = model(img_path, return_usage=True)
        except Exception as e:
            logging.warning(f"[Image one/many+barcode] Error for {img_path}: {e}")
            pred = "unknown|unknown"
            usage = {
                "prompt_token_count": None,
                "candidates_token_count": None,
                "total_token_count": None,
            }
        predictions.append(pred)
        token_stats.append(usage)
        if pred.lower().startswith("many"):
            break
    return predictions, token_stats


# --- Класс-инференс для работы с промптом one|many + barcode ---
class GeminiPhotoOneManyBarcodeInference:
    def __init__(self, api_keys, model_name="gemini-2.0-flash-lite"):
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
                "max_output_tokens": 5000,
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
            system_instruction=PHOTO_ONE_MANY_BARCODE_PROMPT,
        )

    def configure_api(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_api()
        logging.info(f"[Photo LLM] Switched to API key index: {self.current_key_index}")

    def get_response(self, img_data, max_retries=5, return_usage=False):
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
                # logging.info(f"[Photo LLM] Full response: {response}")  # убираем подробный лог

                answer_text = None
                usage = None
                if response.candidates:
                    cand = response.candidates[0]
                    if cand.content and cand.content.parts:
                        answer_text = cand.content.parts[0].text
                    # usage_metadata всегда на верхнем уровне ответа
                    if hasattr(response, "result") and hasattr(
                        response.result, "usage_metadata"
                    ):
                        usage = usage_to_dict(response.result.usage_metadata)
                    elif hasattr(response, "usage_metadata"):
                        usage = usage_to_dict(response.usage_metadata)
                    else:
                        usage = usage_to_dict(None)

                if not answer_text:
                    logging.warning("[Photo LLM] Empty answer from model")
                    return (None, usage) if return_usage else None

                return (
                    (answer_text.strip(), usage)
                    if return_usage
                    else answer_text.strip()
                )

            except Exception as e:
                if "quota" in str(e).lower():
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    if delay > 300:
                        self.switch_api_key()
                        delay = base_delay
                    logging.warning(
                        f"[Photo LLM] Rate limit reached. Attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {delay:.2f} sec..."
                    )
                    time.sleep(delay)
                else:
                    logging.error(f"[Photo LLM] Error in get_response: {str(e)}")
                    raise

        logging.error("[Photo LLM] Max retries reached. Unable to get a response.")
        return (
            (
                None,
                {
                    "prompt_token_count": None,
                    "candidates_token_count": None,
                    "total_token_count": None,
                },
            )
            if return_usage
            else None
        )

    def __call__(self, image_path, return_usage=False):
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
            result = self.get_response(img_data, return_usage=return_usage)
            if not result or (return_usage and not result[0]):
                logging.info("[Photo LLM] Empty answer, retrying...")
                continue
            if return_usage:
                answer, usage = result
                logging.info(
                    f"[LLM photo one/many+barcode] Ответ: {answer} | usage: {usage}"
                )
                if any(
                    answer.lower().startswith(x) for x in ("one|", "many|", "unknown|")
                ):
                    return answer, usage
            else:
                answer = result
                logging.info(f"[LLM photo one/many+barcode] Ответ: {answer}")
                if any(
                    answer.lower().startswith(x) for x in ("one|", "many|", "unknown|")
                ):
                    return answer
            logging.info(f"[Photo LLM] Invalid answer '{answer}', retrying...")

        logging.warning(
            "[Photo LLM] All attempts failed or only invalid answers found."
        )
        if return_usage:
            return "unknown|unknown", {
                "prompt_token_count": None,
                "candidates_token_count": None,
                "total_token_count": None,
            }
        return "unknown|unknown"  # fallback

from dataclasses import asdict, is_dataclass
import google.generativeai as genai
from pathlib import Path
import random
import logging
import time
import json
import os
from PIL import Image
import requests
import re
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_PROMPT = """
You are an expert at extracting automotive part/model numbers from images. You are given the following from a previous description analysis:

[Description LLM output]
Brand | Numbers | One_or_many: {car_brand}

Instructions:
- Use all info above to help analyze the image. If a field is missing or 'None', try to fill it from the image.
- Prefer numbers from the description if they match the typical format for the detected brand (see table below), even if the image is unclear or ambiguous.
- If the image and description disagree, but the description matches the brand's format, trust the description.
- Use the table below to validate and extract the most likely OEM number. If both description and image provide numbers, but only the description matches the format, use the description.
- The main OEM/model number is usually 9-15 characters, contains both letters and digits, and is not a date, batch, or short code. It is often near the barcode and brand logo, in a larger or bolder font.
- Ignore numbers that are clearly dates, batches, or do not match the brand's typical format.
- If there are several numbers, prefer the one closest to the brand/logo/barcode and matching the format.
- If the brand is Bosch, also try to guess for which car brand this part is intended (e.g. Bosch (for Geely)).
- If there is only one physical object in the image, output its main OEM/model number and set the last field to 'one'. If there are multiple objects, output all numbers (comma-separated) and set the last field to 'many'.
- If unsure, default to 'one'.

# OEM Numbers for Car Computers (ECU/ECM/PCM)
| Brand                | Typical Format              | Examples                                      |
|----------------------|----------------------------|-----------------------------------------------|
| Toyota / Lexus       | 10 digits with dash        | 89661-02K21, 89661-0D110, 89661-60A30         |
| Nissan               | Mix of letters & digits    | MEC32-560 A1, A18-000 M42, 23710-8H80A        |
| Ford                 | Letters + digits + suffix  | 98AB-12A650-AD, 2S7A-12A650-LB, F7TF-12A650-BB|
| BMW                  | 7-digit code (Bosch/Siemens)| 7533930, 7613572, 7857277                    |
| Mercedes-Benz        | A + 10 digits              | A 271 153 56 79, A 642 150 26 79, A 646 150 79 79 |
| Volkswagen / Audi    | Digits + letters + suffix  | 06A 906 032 HF, 03G 906 021 LG, 8E0 909 518 AF|
| Hyundai / Kia        | 11 digits with dash        | 39110-2B103, 39100-2A960, 39101-2B020         |
| Honda                | 37820-XXXXX-XXX            | 37820-RNA-A01, 37820-PND-A51, 37820-RBB-A04   |
| Geely (China)        | 10 digits (1016… series)   | 1016051166, 1016055687, 1016057314            |
| Chery (China)        | Alphanumeric with dash     | A11-3605010BB, T11-3605010, A21-3605010BA     |
| Great Wall / Haval (China)| 3605… / 36051… codes  | 3605100-EG01, 3605100-K00, 3605100XKZ16A      |

Output strictly in this format (always in English, always 3 fields, always separated by |):
[Brand/Model Guess] | [Model/Part Number(s)] | [one/many]

If you don't know a value, write None. Do not output anything else except the required 3 fields in the specified format. Always answer in English.
"""


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


class GeminiInference:
    def __init__(
        self,
        api_keys,
        model_name="gemini-2.5-flash",
        car_brand=None,
        prompt_override=None,
    ):
        logging.info(f"[GeminiInference] Using model: {model_name}")

        self.car_brand = car_brand

        self.api_keys = api_keys
        self.current_key_index = 0
        prompt_filled = DEFAULT_PROMPT.format(
            car_brand=car_brand if car_brand is not None else "None"
        )
        if prompt_override:
            self.system_prompt = prompt_override.strip() + "\n\n" + prompt_filled
        else:
            self.system_prompt = prompt_filled

        self.configure_api()
        generation_config = {
            "temperature": 1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 8192,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]

        # Всегда используем только DEFAULT_PROMPT

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=self.system_prompt,
        )

        self.validator_model = self.create_validator_model(model_name)
        self.incorrect_predictions = []
        self.message_history = []

    def load_prompts(self):
        try:
            with open("prompts.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("prompts.json not found. Using default prompts.")
            return {}

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
            system_instruction=OEM_MODEL_PROMPT,
        )

    def switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_api()
        logging.info(f"Switched to API key index: {self.current_key_index}")

    def create_validator_model(self, model_name):
        genai.configure(api_key=self.api_keys[self.current_key_index])

        generation_config = {
            "temperature": 1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 8192,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]
        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    def get_response(self, img_data, retry=False, return_usage=False):
        max_retries = 10
        base_delay = 5

        for attempt in range(max_retries):
            try:
                image_part = {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": (
                            img_data.getvalue()
                            if isinstance(img_data, io.BytesIO)
                            else img_data.read_bytes()
                        ),
                    }
                }
                prompt_parts = [self.system_prompt, image_part]
                if retry:
                    prompt_parts.append(
                        "It is not correct. Try again. Look for the numbers that are highly OEM number"
                    )

                time.sleep(random.uniform(1, 3))

                chat = self.model.start_chat(history=self.message_history)
                response = chat.send_message(prompt_parts)

                # logging.info(f"Main model response: {response.text}")

                self.message_history.append({"role": "user", "parts": prompt_parts})
                self.message_history.append({"role": "model", "parts": [response.text]})

                # usage_metadata может быть на response или response.result
                usage = None
                if hasattr(response, "result") and hasattr(
                    response.result, "usage_metadata"
                ):
                    usage = usage_to_dict(response.result.usage_metadata)
                elif hasattr(response, "usage_metadata"):
                    usage = usage_to_dict(response.usage_metadata)
                else:
                    usage = usage_to_dict(None)

                return (response.text, usage) if return_usage else response.text

            except Exception as e:
                if "quota" in str(e).lower():
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    if delay > 300:
                        self.switch_api_key()
                        delay = base_delay
                    logging.warning(
                        f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logging.error(f"Error in get_response: {str(e)}")
                    raise

        logging.error("Max retries reached. Unable to get a response.")
        raise Exception("Max retries reached. Unable to get a response.")

    def format_part_number(self, number):
        if self.car_brand == "audi" and re.match(
            r"^[A-Z0-9]{3}[0-9]{3}[0-9]{3,5}[A-Z]?$",
            number.replace(" ", "").replace("-", ""),
        ):
            number = number.replace("-", "").replace(" ", "")

            formatted_number = f"{number[:3]} {number[3:6]} {number[6:9]}"

            if len(number) > 9:
                formatted_number += f" {number[9:]}"

            return formatted_number.strip()
        else:
            return number

    def extract_number(self, response):
        number = response.split("<START>")[-1].split("<END>")[0].strip()
        # Replace all 'nan' with 'None' in the output for consistency
        number = number.replace("nan", "None")
        if number.upper() != "NONE":
            return self.format_part_number(number)
        return number

    def validate_number(self, extracted_number, img_data):
        genai.configure(api_key=self.api_keys[self.current_key_index])

        formatted_number = self.format_part_number(extracted_number)

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

        validation_prompt = self.prompts.get(self.car_brand, {}).get(
            "validation_prompt", ""
        )
        incorrect_predictions_str = ", ".join(self.incorrect_predictions)
        prompt = validation_prompt.format(
            extracted_number=extracted_number,
            incorrect_predictions=incorrect_predictions_str,
        )

        prompt_parts = [
            image_parts[0],
            prompt,
        ]

        response = self.validator_model.generate_content(prompt_parts)

        logging.info(f"Validator model response: {response.text}")
        return response.text

    def reset_incorrect_predictions(self):
        self.incorrect_predictions = []
        self.message_history = []

    def __call__(self, image_path, return_usage=False):
        self.configure_api()

        if image_path.startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0 (compatible; autoparts-bot/1.0)"}
            response = requests.get(image_path, stream=True, headers=headers)
            img_data = io.BytesIO(response.content)
        else:
            img = Path(image_path)
            if not img.exists():
                raise FileNotFoundError(f"Could not find image: {img}")
            img_data = img

        self.message_history = []
        num_keys = len(self.api_keys)
        max_attempts = 2
        for key_attempt in range(num_keys):
            self.current_key_index = key_attempt
            self.configure_api()
            logging.info(
                f"[GeminiInference] Using API key index {self.current_key_index}: {self.api_keys[self.current_key_index]}"
            )
            for attempt in range(max_attempts):
                if attempt == 1:
                    orig_prompt = self.system_prompt
                    self.system_prompt = (
                        "Previous answer did not match required format (must contain exactly 2 pipe | characters and 3 fields). STRICTLY follow the output format!\n\n"
                        + orig_prompt
                    )
                result = self.get_response(
                    img_data, retry=(attempt > 0), return_usage=return_usage
                )
                if return_usage:
                    answer, usage = result
                    if not isinstance(usage, dict):
                        try:
                            from dataclasses import asdict

                            usage = asdict(usage)
                        except Exception:
                            usage = (
                                vars(usage)
                                if hasattr(usage, "__dict__")
                                else dict(usage)
                            )
                    if not answer or not isinstance(answer, str):
                        continue
                    if answer.count("|") == 2:
                        logging.info(f"[GeminiInference] Answer: {answer}")
                        return answer, usage
                else:
                    answer = result
                    if not answer or not isinstance(answer, str):
                        continue
                    if answer.count("|") == 2:
                        logging.info(f"[GeminiInference] Answer: {answer}")
                        return answer
            logging.info(
                f"[GeminiInference] Switching to next API key (index {key_attempt+1})"
            )
        raise Exception("Max attempts reached for GeminiInference.")

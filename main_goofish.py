from config import *
from picker_model import TargetModel
from gemini_model import GeminiInference
from dataprocessor_goofish import Processor

import argparse

import pandas as pd
import pickle


import logging
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

cfg = Config
logs = Logs()


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running the Extra")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model to use, e.g., 'gemini'",
    )
    parser.add_argument(
        "--api-keys", nargs="+", required=True, help="List of API keys to use"
    )
    parser.add_argument(
        "--gemini-api-model",
        type=str,
        default="gemini-2.5-pro",
        required=False,
        help="Gemini model u going to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=False,
        help="source to txt file write prompt written inside",
    )
    parser.add_argument(
        "--first-page-link", type=str, default=None, required=False, help=""
    )  # Made optional
    parser.add_argument(
        "--save-file-name", type=str, default="recognized_data", required=False, help=""
    )
    parser.add_argument(
        "--ignore-error",
        action="store_true",
        help="Ignore errors and continue processing",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        required=False,
        help="Maximum steps to collect links",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=90,
        required=False,
        help="Maximum number of links to collect",
    )
    parser.add_argument(
        "--car-brand",
        type=str,
        required=True,
        help="Car brand to use for prompts. Supported brands: audi, toyota, nissan, suzuki, honda, daihatsu, subaru, mazda, bmw, lexus, volkswagen, volvo, mini, fiat, citroen, renault, ford, isuzu, opel, mitsubishi, mercedes, jaguar, peugeot, porsche, alfa_romeo, chevrolet",
    )
    parser.add_argument(
        "--prompt_override",
        type=str,
        default=None,
        help="Override system prompt for Gemini",
    )
    parser.add_argument(
        "--search_keyword",
        type=str,
        default=None,
        help="Keyword for Yahoo search (e.g., ABS)",
    )

    args = parser.parse_args()

    first_page_link = args.first_page_link or cfg.mainpage_url_goofish

    if args.prompt is None:
        prompt = None
    else:
        try:
            print(f"Reading file in {args.prompt}")
            with open(args.prompt, "r") as f:
                prompt = f.read()
                print("Readed Prompt: ", prompt)
        except Exception as e:
            print(f"Error while reading the {args.prompt}:", e)
            prompt = None

    return (
        args.model,
        args.api_keys,
        {
            "gemini_model": args.gemini_api_model,
            "prompt": prompt,
            "main_link": first_page_link,
            "savename": args.save_file_name,
            "ignore_error": args.ignore_error,
            "max_steps": args.max_steps,
            "max_links": args.max_links,
            "car_brand": args.car_brand,
            "prompt_override": args.prompt_override,
        },
    )



def encode(
    link: str, picker: TargetModel, model: GeminiInference, processor, **kwargs
) -> dict:
    logging.info(f"Processing link: {link}")
    max_retries = 3
    base_delay = 5

    for attempt in range(max_retries):
        try:
            page_img_links = processor.parse_big_images_from_slider_selenium(link)
            page_img_links = list(set(page_img_links))

            logging.info(f"Checking {link} for images")
            logging.info(f"Found {len(page_img_links)} unique image links")

            if not page_img_links:
                logging.warning(f"No images found for link: {link}")
                print("DEBUG page_img_links:", page_img_links)
                print("DEBUG target_image_link:", target_image_link)
                print(
                    "DEBUG incorrect_image_links:",
                    [l for l in page_img_links if l != target_image_link],
                )
                return {
                    "predicted_number": "NO_IMAGES",
                    "url": link,
                    "price": "N/A",
                    "correct_image_link": "N/A",
                    "incorrect_image_links": ["N/A"],
                }

            try:
                images_probs = picker.do_inference_return_probs(page_img_links)
            except ValueError as ve:
                if "math domain error" in str(ve).lower():
                    logging.warning(
                        f"Math domain error occurred during inference. Using default probabilities."
                    )
                    images_probs = [
                        {"image_link": link, "score": 1.0 / len(page_img_links)}
                        for link in page_img_links
                    ]
                else:
                    raise

            detail_number = "none"
            target_image_link = None

            for target_image_link, score in [
                (i["image_link"], i["score"]) for i in images_probs
            ]:
                try:
                    logging.info(
                        f"Predicting on image {target_image_link} with score {score}"
                    )
                    detail_number = str(model(target_image_link))

                    if detail_number.lower().strip() != "none":
                        break
                except Exception as e:
                    if "429 Resource has been exhausted" in str(e):
                        delay = base_delay + random.uniform(0, 2)
                        logging.warning(
                            f"429 error encountered. Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                        continue
                    logging.warning(f"Error processing image {target_image_link}: {e}")
                    continue

            if detail_number.lower().strip() == "none":
                logging.warning("No detail number found in any image")

            logging.info(f"Predicted number id: {detail_number}")

            parsed_info = processor.load_product_info_selenium(link)
            return {
                "predicted_number": detail_number,
                "url": link,
                "price": parsed_info.get("price", "N/A"),
                "correct_image_link": target_image_link,
                "incorrect_image_links": [
                    l for l in page_img_links if l != target_image_link
                ]
                or ["N/A"],
            }
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay + random.uniform(0, 2)
                logging.warning(
                    f"Error occurred: {e}. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            else:
                logging.error(
                    f"Error processing link {link} after {max_retries} attempts: {e}"
                )
                return {
                    "predicted_number": "ERROR",
                    "url": link,
                    "price": "N/A",
                    "correct_image_link": "N/A",
                    "incorrect_image_links": ["N/A"],
                }


def save_intermediate_results(result, filename):
    try:
        pd.DataFrame(result).to_excel(f"{filename}.xlsx", index=False)
        logging.info(f"Intermediate results saved to {filename}.xlsx")
    except Exception as e:
        logging.error(
            f"Error saving intermediate results to Excel: {e}. Saving in pickle format instead."
        )
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(result, f)


def reduce(
    main_link: str,
    picker: TargetModel,
    model: GeminiInference,
    processor: Processor,
    ignore_error: bool = False,
    max_steps: int = 3,
    max_links: int = 90,
    savename: str = "recognized_data",
    **kwargs,
):
    logging.info(f"Starting link collection from {main_link}")
    all_links = processor.collect_product_links_selenium(
        max_steps=additional_data["max_steps"], max_links=additional_data["max_links"]
    )
    all_links = list(set([l[1] for l in all_links]))  # l[1] — ссылка на карточку
    logging.info(f"Collected {len(all_links)} unique links")

    result = {
        "predicted_number": [],
        "url": [],
        "price": [],
        "correct_image_link": [],
        "incorrect_image_links": [],
    }

    max_retries = 20
    base_delay = 5

    for i, page_link in enumerate(all_links):
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(1, 3))
                logging.info(f"Processing {i+1}/{len(all_links)} link: {page_link}")
                encoded_data = encode(page_link, picker, model, processor)
                processor.process_encoded_data(encoded_data, result)
                if (i + 1) % 10 == 0:
                    save_intermediate_results(result, f"{savename}_part_{i // 10 + 1}")
                logging.info("Processing successful")
                break
            except Exception as e:
                logging.error(f"Unexpected error processing link {page_link}: {e}")
                encoded_data = {
                    "predicted_number": "ERROR",
                    "url": page_link,
                    "price": "N/A",
                    "correct_image_link": "N/A",
                    "incorrect_image_links": ["N/A"],
                }
                processor.process_encoded_data(encoded_data, result)
                break

    return result


if __name__ == "__main__":
    model_name, api_keys, additional_data = parse_args()
    assert model_name in ["gemini"], "There is no available model you're looking for"

    if model_name == "gemini":
        model = GeminiInference(
            api_keys=api_keys,
            model_name=additional_data["gemini_model"],
            car_brand=additional_data["car_brand"],
            prompt_override=additional_data["prompt_override"],
        )
    else:
        model = None

    processor = Processor(image_size=(512, 512), batch_size=32)
    picker = TargetModel()

    logging.info(f"Starting encoding process with model: {model_name}")
    encoding_result = reduce(
        additional_data["main_link"],
        picker=picker,
        model=model,
        processor=processor,
        ignore_error=additional_data["ignore_error"],
        max_steps=additional_data["max_steps"],
        max_links=additional_data["max_links"],
        savename=additional_data["savename"],
    )

    # Save final results
    try:
        pd.DataFrame(encoding_result).to_excel(
            f"{additional_data['savename']}.xlsx", index=False
        )
        logging.info(f"Final results saved to {additional_data['savename']}.xlsx")
    except Exception as e:
        logging.error(f"Error saving to Excel: {e}. Saving in pickle format instead.")
        with open(f'{additional_data["savename"]}.pkl', "wb") as f:
            pickle.dump(encoding_result, f)
        logging.info(f"Final results saved to {additional_data['savename']}.pkl")

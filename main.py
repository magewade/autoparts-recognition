from config import *
from picker_model import TargetModel
from gemini_model import GeminiInference
from collect_data import collect_links

import argparse

import telebot
import numpy as np
import pandas as pd
import pickle
import requests
import json

from io import BytesIO
from PIL import Image
from IPython.display import clear_output

import logging
import time
import random
from requests.exceptions import RequestException
import urllib.parse
import re

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
        default="gemini-1.5-pro",
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

    # Load prompts.json to get the default first page URL
    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    # Use provided URL or get from prompts.json based on car brand
    first_page_link = args.first_page_link
    if first_page_link is None:
        if args.car_brand.lower() in prompts:
            # Получаем номер категории из prompts.json
            base_url = prompts[args.car_brand.lower()]["first_page_url"]
            m = re.search(r"auccat=(\d+)", base_url)
            if m:
                auccat = m.group(1)
            else:
                raise ValueError(
                    "Can't find category id (auccat) in prompts.json link!"
                )

            # Формируем поисковую ссылку
            search_keyword = args.search_keyword or ""
            # Склеиваем бренд и ключевое слово через +, если оба есть
            if search_keyword:
                if args.car_brand:
                    p_value = f"{args.car_brand} {search_keyword}"
                else:
                    p_value = search_keyword
            else:
                p_value = args.car_brand

            p_value = urllib.parse.quote(p_value)
            first_page_link = (
                f"https://auctions.yahoo.co.jp/search/search?"
                f"istatus=2&is_postage_mode=1&dest_pref_code=13&b=1&n=100&s1=new&o1=d"
                f"&user_type=c&auccat={auccat}&tab_ex=commerce&ei=utf-8&aq=-1&oq=&sc_i=&p={p_value}&x=0&y=0"
            )
            logging.info(
                f"Using Yahoo search URL with category and keyword: {first_page_link}"
            )
        else:
            raise ValueError(f"Car brand '{args.car_brand}' not found in prompts.json")

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
            "main_link": first_page_link,  # Use the determined first page link
            "savename": args.save_file_name,
            "ignore_error": args.ignore_error,
            "max_steps": args.max_steps,
            "max_links": args.max_links,
            "car_brand": args.car_brand,
            "prompt_override": args.prompt_override,
        },
    )


import math


def encode(link: str, picker: TargetModel, model: GeminiInference, **kwargs) -> dict:
    logging.info(f"Processing link: {link}")
    max_retries = 3
    base_delay = 5

    for attempt in range(max_retries):
        try:
            page_img_links = picker.processor.parse_images_from_page(link)
            page_img_links = list(set(page_img_links))

            logging.info(f"Checking {link} for images")
            logging.info(f"Found {len(page_img_links)} unique image links")

            if not page_img_links:
                logging.warning(f"No images found for link: {link}")
                return {
                    "predicted_number": "NO_IMAGES",
                    "url": link,
                    "price": "N/A",
                    "correct_image_link": "N/A",
                    "incorrect_image_links": "N/A",
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

            parsed_info = picker.processor.load_product_info(link)
            return {
                "predicted_number": detail_number,
                "url": link,
                "price": parsed_info.get("price", "N/A"),
                "correct_image_link": target_image_link,
                "incorrect_image_links": ", ".join(
                    [l for l in page_img_links if l != target_image_link]
                ),
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
                    "incorrect_image_links": "N/A",
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
    model: GeminiInference,  # Add model as a parameter
    ignore_error: bool = False,
    max_steps: int = 3,
    max_links: int = 90,
    savename: str = "recognized_data",
    **kwargs,
):
    logging.info(f"Starting link collection from {main_link}")
    all_links = collect_links(
        picker, main_link, max_pages=max_steps, max_links=max_links
    )
    all_links = list(set(all_links))
    logging.info(f"Collected {len(all_links)} unique links")

    result = {
        "predicted_number": list(),
        "url": list(),
        "price": list(),
        "correct_image_link": list(),
        "incorrect_image_links": list(),
    }

    max_retries = 20
    base_delay = 5  # Initial delay in seconds

    for i, page_link in enumerate(all_links):
        for attempt in range(max_retries):
            try:
                # Add a small random delay before each request
                time.sleep(random.uniform(1, 3))

                logging.info(f"Processing {i+1}/{len(all_links)} link: {page_link}")
                encoded_data = encode(page_link, picker, model)  # Remove kwargs here
                for k, v in encoded_data.items():
                    result[k].append(v)

                if (i + 1) % 10 == 0:  # Save every 10 iterations
                    save_intermediate_results(result, f"{savename}_part_{i // 10 + 1}")

                logging.info("Processing successful")
                break  # If successful, break out of the retry loop

            except Exception as e:
                logging.error(f"Unexpected error processing link {page_link}: {e}")
                if not ignore_error:
                    logging.error("Stopping due to error and ignore_error=False")
                    return result
                logging.warning("Ignoring error and moving to next link")
                break  # Move to next link if ignore_error is True

    return result


if __name__ == "__main__":
    # Parse important variables
    model_name, api_keys, additional_data = parse_args()

    # Initialize models
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

    picker = TargetModel()

    logging.info(f"Starting encoding process with model: {model_name}")
    encoding_result = reduce(
        additional_data["main_link"],
        picker=picker,
        model=model,
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

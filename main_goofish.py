from config import *
from picker_model import TargetModel
from gemini_model import GeminiInference
from dataprocessor_goofish import Processor

import argparse

import pandas as pd
import os
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
    url: str, images: list, price: str, picker: TargetModel, model: GeminiInference
) -> dict:
    logging.info(f"Processing url: {url}")
    logging.info(f"Найдено картинок: {len(images)}")
    if not images or images == [""]:
        return {
            "predicted_number": "NO_IMAGES",
            "url": url,
            "price": price,
            "correct_image_link": "N/A",
            "incorrect_image_links": ["N/A"],
        }
    try:
        images_probs = picker.do_inference_return_probs(images)
    except Exception as e:
        logging.warning(f"Error in do_inference_return_probs: {e}")
        images_probs = [
            {"image_link": img, "score": 1.0 / len(images)} for img in images
        ]
    # Логируем уверенность для каждой картинки
    for i in images_probs:
        logging.info(
            f"Картинка: {i['image_link']} — уверенность: {i['score']*100:.1f}%"
        )
    detail_number = "none"
    target_image_link = None
    for target_image_link, score in [
        (i["image_link"], i["score"]) for i in images_probs
    ]:
        try:
            logging.info(
                f"Модель предсказывает по картинке: {target_image_link} (уверенность {score*100:.1f}%)"
            )
            detail_number = str(model(target_image_link))
            logging.info(
                f"Картинка {target_image_link} — предсказан номер: {detail_number}"
            )
            if detail_number.lower().strip() != "none":
                break
        except Exception as e:
            logging.warning(f"Error processing image {target_image_link}: {e}")
            continue
    if detail_number.lower().strip() == "none":
        logging.warning("No detail number found in any image")
    return {
        "predicted_number": detail_number,
        "url": url,
        "price": price,
        "correct_image_link": target_image_link,
        "incorrect_image_links": [l for l in images if l != target_image_link]
        or ["N/A"],
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
    picker: TargetModel,
    model: GeminiInference,
    savename: str = "recognized_data",
    parsed_csv: str = "parsed_products.csv",
):
    df = pd.read_csv(parsed_csv)
    result = {
        "predicted_number": [],
        "url": [],
        "price": [],
        "correct_image_link": [],
        "incorrect_image_links": [],
    }
    # Проверяем, есть ли уже сохранённый результат
    processed_urls = set()
    import os

    if os.path.exists(f"{savename}.xlsx"):
        try:
            prev = pd.read_excel(f"{savename}.xlsx")
            processed_urls = set(prev["url"].astype(str))
            # Восстанавливаем результат, чтобы не потерять старые строки
            for col in result:
                if col in prev:
                    result[col] = prev[col].tolist()
        except Exception as e:
            logging.warning(f"Could not load previous results: {e}")
    # Обрабатываем только новые строки
    for i, row in df.iterrows():
        url = str(row.get("url", ""))
        if url in processed_urls:
            continue
        price = row.get("price", "N/A")
        logging.info(f"Найдена цена: {price}")
        images = row.get("images", "")
        images_list = [img for img in str(images).split(",") if img]
        try:
            encoded_data = encode(url, images_list, price, picker, model)
        except Exception as e:
            logging.error(f"Error processing row {i} ({url}): {e}")
            encoded_data = {
                "predicted_number": "ERROR",
                "url": url,
                "price": price,
                "correct_image_link": "N/A",
                "incorrect_image_links": images_list or ["N/A"],
            }
        for k, v in encoded_data.items():
            if k == "incorrect_image_links":
                result[k].append(", ".join(v) if isinstance(v, list) else str(v))
            else:
                result[k].append(v)
        if (len(result["url"]) % 10) == 0:
            save_intermediate_results(
                result, f"{savename}_part_{len(result['url']) // 10}"
            )
        # Сохраняем прогресс после каждой строки (можно закомментировать для ускорения)
        try:
            pd.DataFrame(result).to_excel(f"{savename}.xlsx", index=False)
        except Exception as e:
            logging.warning(f"Could not save progress: {e}")

    return result


if __name__ == "__main__":

    # 1. Сбор ссылок — product_links.csv
    _, _, additional_data = parse_args()
    processor = Processor(image_size=(512, 512), batch_size=32)
    driver = processor.create_persistent_driver()
    try:
        links = processor.collect_product_links_selenium(
            driver,
            max_steps=additional_data["max_steps"],
            max_links=additional_data["max_links"],
        )
        processor.save_product_links_to_csv(links, filename="product_links.csv")
    finally:
        processor.close_persistent_driver(driver)

    # 2. Playwright: product_links.csv -> parsed_products.csv
    import pandas as pd
    import asyncio
    from dataprocessor_goofish import GoofishParserPlaywrightAsync

    async def process_links_playwright(
        links, batch_size=10, output_csv="parsed_products.csv"
    ):
        parser = GoofishParserPlaywrightAsync()
        from playwright.async_api import async_playwright
        import csv

        if os.path.exists(output_csv):
            os.remove(output_csv)
        fieldnames = ["url", "images", "price"]
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            for i in range(0, len(links), batch_size):
                batch = links[i : i + batch_size]
                results = []
                for url in batch:
                    try:
                        images = await parser.parse_big_images(url, page)
                        info = await parser.load_product_info(url, page)
                        results.append(
                            {
                                "url": url,
                                "images": ",".join(images),
                                "price": info.get("price", "N/A"),
                            }
                        )
                    except Exception as e:
                        results.append(
                            {"url": url, "images": "ERROR", "price": "ERROR"}
                        )
                with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    for row in results:
                        writer.writerow(row)
            await browser.close()

    df_links = pd.read_csv("product_links.csv")
    card_links = df_links["href"].dropna().unique().tolist()
    asyncio.run(
        process_links_playwright(
            card_links, batch_size=10, output_csv="parsed_products.csv"
        )
    )

    # 3. Инференс: parsed_products.csv -> final_products.csv
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
    # Читаем parsed_products.csv, добавляем новые колонки и сохраняем final_products.csv
    df = pd.read_csv("parsed_products.csv")
    new_cols = ["predicted_number", "correct_image_link"]
    df[new_cols[0]] = ""
    df[new_cols[1]] = ""
    for i, row in df.iterrows():
        url = row.get("url", "")
        price = row.get("price", "N/A")
        images = row.get("images", "")
        images_list = [img for img in str(images).split(",") if img]
        try:
            encoded_data = encode(url, images_list, price, picker, model)
        except Exception as e:
            logging.error(f"Error processing row {i} ({url}): {e}")
            encoded_data = {
                "predicted_number": "ERROR",
                "correct_image_link": "N/A",
            }
        df.at[i, "predicted_number"] = encoded_data["predicted_number"]
        df.at[i, "correct_image_link"] = encoded_data["correct_image_link"]
    df.to_csv("final_products.csv", index=False)
    logging.info("Final results saved to final_products.csv")

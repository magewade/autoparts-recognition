import ast

import logging
import pandas as pd
import os
import argparse
from config import Config
from dataprocessor_goofish import (
    Processor,
    GoofishParserPlaywrightAsync,
    enrich_dataframe_playwright_async,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_inference(parsed_csv="parsed_products.csv", output_csv="final_products.csv"):
    from picker_model import TargetModel
    from gemini_model import GeminiInference
    import pandas as pd
    import logging
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-keys", nargs="+", required=True, help="List of API keys to use"
    )
    parser.add_argument(
        "--gemini-api-model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model name",
    )
    parser.add_argument("--car-brand", type=str, default=None, help="Car brand")
    parser.add_argument(
        "--prompt_override", type=str, default=None, help="Prompt override"
    )
    parser.add_argument(
        "--save-file-name",
        type=str,
        default="final_products",
        help="Output CSV base name",
    )
    args, _ = parser.parse_known_args()

    picker = TargetModel()
    llm = GeminiInference(
        api_keys=args.api_keys,
        model_name=args.gemini_api_model,
        car_brand=args.car_brand,
        prompt_override=args.prompt_override,
    )
    df = pd.read_csv(parsed_csv)
    predicted_images = []
    llm_predictions = []
    confidences = []
    for i, row in df.iterrows():
        images = row.get("images", "[]")
        try:
            images_list = (
                ast.literal_eval(images) if isinstance(images, str) else images
            )
        except Exception:
            images_list = []
        if not images_list:
            predicted_images.append("")
            llm_predictions.append("")
            confidences.append("")
            continue
        try:
            pred_img, conf = picker.do_inference(images_list)
            predicted_images.append(pred_img)
            confidences.append(conf)
            logging.info(f"Predicted image: {pred_img} | Confidence: {conf:.4f}")
        except Exception as e:
            logging.warning(f"Picker error: {e}")
            predicted_images.append(images_list[0])
            confidences.append("")
        try:
            llm_pred = llm(predicted_images[-1])
            llm_predictions.append(llm_pred)
        except Exception as e:
            logging.warning(f"LLM error: {e}")
            llm_predictions.append("")
    df["predicted_image"] = predicted_images
    df["confidence"] = confidences
    df["llm_prediction"] = llm_predictions
    df.to_csv(args.save_file_name + ".csv", index=False)
    logging.info(f"Inference results saved to {args.save_file_name}.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Goofish pipeline")
    parser.add_argument("--max-steps", type=int, default=3, help="Max pages to parse")
    parser.add_argument(
        "--max-links", type=int, default=90, help="Max product links to collect"
    )
    # остальные аргументы игнорируем, чтобы не мешали запуску
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    cfg = Config
    processor = Processor(image_size=(512, 512), batch_size=32)
    driver = processor.create_persistent_driver()
    try:
        links = processor.collect_product_links_selenium(
            driver,
            max_steps=args.max_steps,
            max_links=args.max_links,
        )
        # Ограничиваем список ссылок до max_links, если их больше
        if len(links) > args.max_links:
            links = links[: args.max_links]
        processor.save_product_links_to_csv(links, filename="product_links.csv")
    finally:
        processor.close_persistent_driver(driver)

    df_links = pd.read_csv("product_links.csv")
    if "href" in df_links.columns:
        df_for_parse = pd.DataFrame({"href": df_links["href"]})
    elif "url" in df_links.columns:
        df_for_parse = pd.DataFrame({"href": df_links["url"]})
    else:
        raise Exception("No href/url column in product_links.csv")

    parser = GoofishParserPlaywrightAsync()
    import asyncio

    df_result = asyncio.run(
        enrich_dataframe_playwright_async(
            df_for_parse, parser, output_path="parsed_products.csv", chunk_size=10
        )
    )
    logging.info("parsed_products.csv сформирован")


if __name__ == "__main__":
    main()
    run_inference(parsed_csv="parsed_products.csv")

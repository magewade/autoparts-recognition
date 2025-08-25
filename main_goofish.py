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
    result_path = args.save_file_name + ".csv"
    if os.path.exists(result_path):
        df_result = pd.read_csv(result_path)
        processed_mask = df_result.get("predicted_image", "").astype(str).str.len() > 0
        predicted_images = df_result.get("predicted_image", [""] * len(df)).tolist()
        confidences = df_result.get("confidence", [""] * len(df)).tolist()
        llm_predictions = df_result.get("llm_prediction", [""] * len(df)).tolist()
        logging.info(f"Продолжаем инференс с {sum(processed_mask)} обработанных строк")
    else:
        predicted_images = [""] * len(df)
        confidences = [""] * len(df)
        llm_predictions = [""] * len(df)
        processed_mask = pd.Series([False] * len(df))
        chunk_size = 10
        for i, row in df.iterrows():
            if processed_mask[i]:
                continue  # уже обработано
            images = row.get("images", "[]")
            try:
                images_list = (
                    ast.literal_eval(images) if isinstance(images, str) else images
                )
            except Exception:
                images_list = []
            if not images_list:
                predicted_images[i] = ""
                llm_predictions[i] = ""
                confidences[i] = ""
                continue
            try:
                pred_img, conf = picker.do_inference(images_list)
                predicted_images[i] = pred_img
                confidences[i] = conf
                logging.info(f"Predicted image: {pred_img} | Confidence: {conf:.4f}")
            except Exception as e:
                logging.warning(f"Picker error: {e}")
                predicted_images[i] = images_list[0]
                confidences[i] = ""
            try:
                llm_pred = llm(predicted_images[i])
                llm_predictions[i] = llm_pred
            except Exception as e:
                logging.warning(f"LLM error: {e}")
                llm_predictions[i] = ""
            if (i + 1) % chunk_size == 0:
                temp_df = df.copy()
                temp_df["predicted_image"] = predicted_images
                temp_df["confidence"] = confidences
                temp_df["llm_prediction"] = llm_predictions
                temp_df.to_csv(result_path, index=False)
                logging.info(
                    f"[Inference] Промежуточные результаты сохранены в {result_path}"
                )
        # Финальный результат, если не делится на chunk_size
        temp_df = df.copy()
        temp_df["predicted_image"] = predicted_images
        temp_df["confidence"] = confidences
        temp_df["llm_prediction"] = llm_predictions
        temp_df.to_csv(result_path, index=False)
    logging.info(f"[Inference] Финальные результаты сохранены в {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Goofish pipeline")
    parser.add_argument("--max-steps", type=int, default=3, help="Max pages to parse")
    parser.add_argument(
        "--max-links", type=int, default=90, help="Max product links to collect"
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    cfg = Config
    # 1. Сбор product_links.csv (если нет)
    if not os.path.exists("product_links.csv"):
        processor = Processor(image_size=(512, 512), batch_size=32)
        driver = processor.create_persistent_driver()
        try:
            links = processor.collect_product_links_selenium(
                driver,
                max_steps=args.max_steps,
                max_links=args.max_links,
            )
            if len(links) > args.max_links:
                links = links[: args.max_links]
            processor.save_product_links_to_csv(links, filename="product_links.csv")
        finally:
            processor.close_persistent_driver(driver)
        logging.info("product_links.csv сформирован")
    else:
        logging.info("product_links.csv найден, пропускаем сбор ссылок")

    # 2. Сбор parsed_products.csv (если нет или не все ссылки обработаны)
    df_links = pd.read_csv("product_links.csv")
    n_links = len(df_links)
    need_parse = True
    if os.path.exists("parsed_products.csv"):
        df_parsed = pd.read_csv("parsed_products.csv")
        n_parsed = len(df_parsed)
        if n_parsed >= n_links:
            need_parse = False
            logging.info(
                "parsed_products.csv найден и все ссылки обработаны, пропускаем парсинг Playwright"
            )
        else:
            logging.info(
                f"parsed_products.csv найден, обработано {n_parsed} из {n_links} — допарсим оставшиеся"
            )
            if "href" in df_links.columns:
                df_for_parse = pd.DataFrame({"href": df_links["href"]})
            elif "url" in df_links.columns:
                df_for_parse = pd.DataFrame({"href": df_links["url"]})
            else:
                raise Exception("No href/url column in product_links.csv")
            parser = GoofishParserPlaywrightAsync()
            import asyncio

            # Парсим только недостающие строки
            df_for_parse = df_for_parse.iloc[n_parsed:]
            df_result = asyncio.run(
                enrich_dataframe_playwright_async(
                    df_for_parse,
                    parser,
                    output_path="parsed_products.csv",
                    chunk_size=10,
                )
            )
            logging.info("parsed_products.csv дополнен оставшимися строками")
    if need_parse:
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

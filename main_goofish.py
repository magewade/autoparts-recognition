import ast

from gemini_model import GeminiInference
import pandas as pd
import time
import logging

from picker_model import TargetModel
from gemini_model import GeminiInference
import pandas as pd
import logging
import argparse
import time
import re

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


def extract_model_from_description(
    input_csv="parsed_products.csv",
    output_csv="parsed_products_with_model.csv",
    api_keys=None,
    model_name="gemini-2.5-flash-lite",
):
    """
    Для каждой строки вызывает GeminiInference по description, сохраняет результат в description_model_guess.
    """

    if api_keys is None:
        raise ValueError("api_keys must be provided for GeminiInference")
    df = pd.read_csv(input_csv)
    if os.path.exists(output_csv):
        df_out = pd.read_csv(output_csv)
        if "description_model_guess" in df_out.columns and len(df_out) == len(df):
            logging.info(f"{output_csv} найден, пропускаем LLM по описанию")
            return output_csv
    llm = GeminiInference(api_keys=api_keys, model_name=model_name)
    logging.info(f"[LLM desc] Используемая модель: {model_name}")
    guesses = []
    for i, row in df.iterrows():
        desc = str(row.get("description", ""))
        if not desc.strip():
            guesses.append("")
            continue
        prompt = (
            "Extract only the car model name (or a list of models separated by commas) from the description, for which the part is intended. "
            "Always answer in English. Output ONLY the car model name(s) in English, no tags, no extra text, no markup, no translation of brand/model names. "
            "If multiple models are mentioned, list them separated by commas. If no model is specified, output exactly: NONE. "
            "Do not use any tags, brackets, or special formatting. Only output the model name(s) separated by commas, or NONE. "
            + desc
        )
        try:
            guess = llm.model.generate_content(prompt).text.strip()
            logging.info(f"[LLM desc] Строка {i}: результат Gemini: {guess}")
        except Exception as e:
            logging.warning(f"Gemini LLM error on row {i}: {e}")
            guess = "ERROR"
        guesses.append(guess)
        if (i + 1) % 10 == 0:
            df_temp = df.copy()
            df_temp["description_model_guess"] = guesses + [""] * (
                len(df) - len(guesses)
            )
            df_temp.to_csv(output_csv, index=False)
            logging.info(f"[LLM desc] Промежуточные результаты записаны в {output_csv}")
        time.sleep(1.5)
    df["description_model_guess"] = guesses
    df.to_csv(output_csv, index=False)
    logging.info(f"[LLM desc] Финальные результаты сохранены в {output_csv}")
    return output_csv


def run_inference(parsed_csv="parsed_products.csv", output_csv="final_products.csv"):
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
    df = pd.read_csv(parsed_csv)
    result_path = args.save_file_name + ".csv"
    # Если есть столбец description_model_guess, будем использовать его для car_brand
    has_model_guess = "description_model_guess" in df.columns
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
            llm_pred = ""
            # --- Передаем модель авто из description_model_guess, если есть ---
            car_brand = args.car_brand
            if has_model_guess:
                guess = str(row.get("description_model_guess", "")).strip()
                if guess and guess.upper() != "NONE":
                    car_brand = guess
            llm_row = GeminiInference(
                api_keys=args.api_keys,
                model_name=args.gemini_api_model,
                car_brand=car_brand,
                prompt_override=args.prompt_override,
            )
            for attempt in range(2):
                try:
                    llm_pred = llm_row(predicted_images[i])
                    if not llm_pred:
                        raise ValueError(f"Empty LLM response: {llm_pred}")
                    break  # успех
                except Exception as e:
                    logging.warning(f"LLM error (attempt {attempt+1}): {e}")
                    llm_pred = ""
            llm_predictions[i] = llm_pred
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
    times = {}
    runtime_logs = []

    orig_logging_info = logging.info

    def custom_logging_info(msg, *a, **kw):
        orig_logging_info(msg, *a, **kw)
        if isinstance(msg, str) and msg.startswith("Runtime of "):
            runtime_logs.append(msg)

    logging.info = custom_logging_info

    # 1. Сбор product_links.csv (если нет)
    if not os.path.exists("product_links.csv"):
        t0 = time.time()
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
        times["collect_links"] = time.time() - t0
        logging.info(f"Время сбора ссылок: {times['collect_links']:.2f} сек")
    else:
        logging.info("product_links.csv найден, пропускаем сбор ссылок")
        times["collect_links"] = None

    # 2. Сбор parsed_products.csv (если нет или не все ссылки обработаны)
    t1 = time.time()
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
            times["parsing"] = None
        else:
            logging.info(
                f"parsed_products.csv найден, обработано {n_parsed} из {n_links}"
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
            t1 = time.time()
            df_for_parse = df_for_parse.iloc[n_parsed:]
            df_result = asyncio.run(
                enrich_dataframe_playwright_async(
                    df_for_parse,
                    parser,
                    output_path="parsed_products.csv",
                    chunk_size=10,
                )
            )
            times["parsing"] = time.time() - t1
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

        t1 = time.time()
        df_result = asyncio.run(
            enrich_dataframe_playwright_async(
                df_for_parse, parser, output_path="parsed_products.csv", chunk_size=10
            )
        )
        times["parsing"] = time.time() - t1
        logging.info("parsed_products.csv сформирован")
    # Восстанавливаем logging.info
    logging.info = orig_logging_info
    return runtime_logs, times


if __name__ == "__main__":
    import time

    main_start = time.time()
    runtime_logs, times = main()

    parsed_csv = "parsed_products.csv"
    parsed_with_model_csv = "parsed_products_with_model.csv"
    output_csv = "final_products.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-keys", nargs="+", required=True, help="List of API keys to use"
    )
    parser.add_argument(
        "--gemini-api-model",
        type=str,
        default="gemini-2.5-flash",
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
    parser.add_argument("--max-steps", type=int, default=3, help="Max pages to parse")
    parser.add_argument(
        "--max-links", type=int, default=90, help="Max product links to collect"
    )
    parser.add_argument(
        "--desc-api-keys",
        nargs="+",
        default=None,
        help="API keys for description LLM (optional)",
    )
    parser.add_argument(
        "--desc-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for description LLM",
    )
    cli_args, _ = parser.parse_known_args()

    api_keys = cli_args.api_keys
    desc_model = cli_args.desc_model
    if cli_args.desc_api_keys is not None:
        api_keys_desc = cli_args.desc_api_keys
    else:
        api_keys_desc = api_keys

    if os.path.exists(parsed_csv):
        # 1. LLM по описанию
        parsed_with_model_csv = extract_model_from_description(
            parsed_csv,
            parsed_with_model_csv,
            api_keys=api_keys_desc,
            model_name=desc_model,
        )
        df_parsed = pd.read_csv(parsed_with_model_csv)
        n_parsed = len(df_parsed)
        if os.path.exists(output_csv):
            df_final = pd.read_csv(output_csv)
            n_final = len(df_final)
            if n_final >= n_parsed:
                logging.info(
                    "final_products.csv найден и все строки обработаны, пропускаем инференс"
                )
                inference_time = None
            else:
                logging.info(
                    f"final_products.csv найден, обработано {n_final} из {n_parsed} — доинференсим оставшиеся"
                )
                t_inf = time.time()
                run_inference(parsed_csv=parsed_with_model_csv, output_csv=output_csv)
                inference_time = time.time() - t_inf
        else:
            t_inf = time.time()
            run_inference(parsed_csv=parsed_with_model_csv, output_csv=output_csv)
            inference_time = time.time() - t_inf
    else:
        logging.warning("parsed_products.csv не найден, инференс невозможен")
        inference_time = None

    # Финальная сводка по времени
    def fmt_time(val):
        return f"{val:.2f} сек" if val is not None else "пропущен"

    total_time = 0.0
    for v in [
        locals().get("times", {}).get("collect_links"),
        locals().get("times", {}).get("parsing"),
        inference_time,
    ]:
        if v is not None:
            total_time += v

    from collections import defaultdict

    times = defaultdict(float, times)  # а не locals().get(...)
    times["inference"] = inference_time
    logging.info("\n==== RUNTIME LOGS ====")
    for log in runtime_logs:
        logging.info(log)
    logging.info(
        "\n==== ВРЕМЯ ЭТАПОВ ===="
        f"\nСбор ссылок: {fmt_time(times['collect_links'])}"
        f"\nПарсинг: {fmt_time(times['parsing'])}"
        f"\nИнференс: {fmt_time(times['inference'])}"
        f"\n----------------------"
        f"\nВсего: {total_time:.2f} сек"
        f"\n====================="
    )

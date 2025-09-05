from gemini_description_model import GeminiDescriptionInference
import os
import ast
import time
import logging
import argparse
import pandas as pd

from config import Config
from gemini_model import GeminiInference
from picker_model import TargetModel
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

    llm = GeminiDescriptionInference(api_keys=api_keys, model_name=model_name)
    logging.info(f"[LLM desc] Используемая модель: {model_name}")
    guesses = []
    for i, row in df.iterrows():
        desc = str(row.get("description", ""))
        if not desc.strip():
            guesses.append("")
            continue
        try:
            guess = llm(desc)
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
            # --- Парсим description_model_guess на brand, numbers, one_many ---
            desc_brand, desc_numbers, desc_one_many = None, None, None
            if has_model_guess:
                guess = str(row.get("description_model_guess", "")).strip()
                if guess and guess.upper() != "NONE":
                    parts = [p.strip() for p in guess.split("|")]
                    if len(parts) == 3:
                        desc_brand, desc_numbers, desc_one_many = parts
            # Если нет guess, используем car_brand из аргументов
            if not desc_brand:
                desc_brand = args.car_brand
            llm_row = GeminiInference(
                api_keys=args.api_keys,
                model_name=args.gemini_api_model,
                car_brand=desc_brand,
                desc_numbers=desc_numbers,
                desc_one_many=desc_one_many,
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
    parser.add_argument(
        "--desc-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for description LLM",
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

    # 1. Сбор product_links.csv (если нет или не хватает ссылок)
    need_collect_links = True
    if os.path.exists("product_links.csv"):
        df_links_check = pd.read_csv("product_links.csv")
        if len(df_links_check) >= args.max_links:
            need_collect_links = False
    if need_collect_links:
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
        logging.info(
            "product_links.csv найден и содержит достаточно ссылок, пропускаем сбор ссылок"
        )
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


def run_full_pipeline(cli_args):
    parsed_csv = "parsed_products.csv"
    parsed_with_model_csv = "parsed_products_with_model.csv"
    output_csv = "final_products.csv"

    api_keys = cli_args.api_keys
    desc_model = cli_args.desc_model
    if cli_args.desc_api_keys is not None:
        api_keys_desc = cli_args.desc_api_keys
    else:
        api_keys_desc = api_keys

    if os.path.exists(parsed_csv):
        parsed_with_model_csv = extract_model_from_description(
            parsed_csv,
            parsed_with_model_csv,
            api_keys=api_keys_desc,
            model_name=desc_model,
        )
        df_parsed = pd.read_csv(parsed_with_model_csv)
        n_parsed = len(df_parsed)

        # --- Разделение на many/one по description_model_guess ---
        def get_one_many(val):
            try:
                return str(val).split("|")[2].strip().lower()
            except Exception:
                return "one"

        mask_many = df_parsed["description_model_guess"].apply(get_one_many) == "many"
        df_many = df_parsed[mask_many].copy()
        df_one = df_parsed[~mask_many].copy()
        df_many.to_csv("products_many.csv", index=False)
        df_one.to_csv("products_one.csv", index=False)
        logging.info(
            f"Сохранено: products_many.csv (many: {len(df_many)}), products_one.csv (one: {len(df_one)})"
        )

        # --- Новый шаг: анализ всех изображений для one (image_predictions) ---
        from gemini_one_many_on_photo import process_images_one_many_and_barcode_label
        import ast

        df_one = df_one.copy()
        image_predictions = []
        for idx, row in df_one.iterrows():
            images = row.get("images", "[]")
            try:
                images_list = (
                    ast.literal_eval(images) if isinstance(images, str) else images
                )
            except Exception:
                images_list = []
            if not images_list:
                image_predictions.append([])
                continue
            try:
                preds = process_images_one_many_and_barcode_label(
                    images_list,
                    api_keys=cli_args.api_keys,
                    model_name=cli_args.gemini_api_model,
                )
            except Exception as e:
                logging.warning(
                    f"[Image one/many+barcode step] Error for row {idx}: {e}"
                )
                preds = ["unknown|unknown"] * len(images_list)
            image_predictions.append(preds)
        df_one["image_predictions"] = image_predictions
        df_one.to_csv("products_one_with_image_analysis.csv", index=False)

        # --- Отсекаем many по image_predictions ---
        def has_many(preds):
            return any(str(p).lower().startswith("many") for p in preds)

        mask_many_img = df_one["image_predictions"].apply(has_many)
        df_many_img = df_one[mask_many_img].copy()
        df_one_img = df_one[~mask_many_img].copy()
        df_many_img.to_csv("products_many_by_image.csv", index=False)
        df_one_img.to_csv("products_one_by_image.csv", index=False)

        # --- Дальнейший пайплайн только для one (products_one_by_image.csv) ---
        one_input_csv = "products_one_by_image.csv"
        if os.path.exists(output_csv):
            df_final = pd.read_csv(output_csv)
            n_final = len(df_final)
            if n_final >= len(df_one_img):
                logging.info(
                    "final_products.csv найден и все строки обработаны, пропускаем инференс"
                )
                inference_time = None
            else:
                logging.info(
                    f"final_products.csv найден, обработано {n_final} из {len(df_one_img)} — доинференсим оставшиеся"
                )
                t_inf = time.time()
                run_inference(parsed_csv=one_input_csv, output_csv=output_csv)
                inference_time = time.time() - t_inf
        else:
            t_inf = time.time()
            run_inference(parsed_csv=one_input_csv, output_csv=output_csv)
            inference_time = time.time() - t_inf
    else:
        logging.warning("parsed_products.csv не найден, инференс невозможен")
        inference_time = None

    return inference_time


if __name__ == "__main__":
    import time

    main_start = time.time()
    runtime_logs, times = main()

    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument(
        "--api-keys", nargs="+", required=True, help="List of API keys to use"
    )
    cli_parser.add_argument(
        "--gemini-api-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    cli_parser.add_argument("--car-brand", type=str, default=None, help="Car brand")
    cli_parser.add_argument(
        "--prompt_override", type=str, default=None, help="Prompt override"
    )
    cli_parser.add_argument(
        "--save-file-name",
        type=str,
        default="final_products",
        help="Output CSV base name",
    )
    cli_parser.add_argument(
        "--max-steps", type=int, default=3, help="Max pages to parse"
    )
    cli_parser.add_argument(
        "--max-links", type=int, default=90, help="Max product links to collect"
    )
    cli_parser.add_argument(
        "--desc-api-keys",
        nargs="+",
        default=None,
        help="API keys for description LLM (optional)",
    )
    cli_parser.add_argument(
        "--desc-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for description LLM",
    )
    cli_args = cli_parser.parse_args()

    # Запуск пайплайна
    inference_time = run_full_pipeline(cli_args)

    # Финальная сводка по времени

    def fmt_time(val):
        return f"{val/60:.2f} мин" if val is not None else "пропущен"

    total_time = 0.0
    for v in [
        locals().get("times", {}).get("collect_links"),
        locals().get("times", {}).get("parsing"),
        inference_time,
    ]:
        if v is not None:
            total_time += v

    from collections import defaultdict

    times = defaultdict(float, times)
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
        f"\nВсего: {total_time/60:.2f} мин"
        f"\n====================="
    )

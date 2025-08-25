# --- SIMPLE TWO-STAGE PIPELINE: 1) LINKS, 2) IMAGES+PRICE ---
import logging
import pandas as pd
import os
from config import Config
from dataprocessor_goofish import (
    Processor,
    GoofishParserPlaywrightAsync,
    enrich_dataframe_playwright_async,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

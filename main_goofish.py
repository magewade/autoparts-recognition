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


def main():
    # 1. Сбор product_links.csv (img_src, href)
    cfg = Config
    processor = Processor(image_size=(512, 512), batch_size=32)
    driver = processor.create_persistent_driver()
    try:
        links = processor.collect_product_links_selenium(
            driver,
            max_steps=3,  # можно вынести в аргументы
            max_links=90,
        )
        processor.save_product_links_to_csv(links, filename="product_links.csv")
    finally:
        processor.close_persistent_driver(driver)

    # 2. enrich_dataframe_playwright_async: по href собирает картинки и цену, сохраняет parsed_products.csv
    df_links = pd.read_csv("product_links.csv")
    # Если в product_links.csv есть только img_src и href, используем href
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

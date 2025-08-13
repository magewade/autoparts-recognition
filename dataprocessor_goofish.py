import logging
from config import Config as cfg
from config import RuntimeMeta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import time

import tensorflow as tf
import numpy as np

import tempfile
import shutil
import os

import random

import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import time
import random


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_chrome_driver(user_agent=None, debug_port=9222):
    chrome_options = Options()
    if user_agent:
        chrome_options.add_argument(f"--user-agent={user_agent}")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"--remote-debugging-port={debug_port}")
    # создаём уникальную временную директорию
    temp_dir = tempfile.mkdtemp(prefix="chrome_user_data_")
    chrome_options.add_argument(f"--user-data-dir={temp_dir}")
    driver = webdriver.Chrome(options=chrome_options)
    driver._temp_dir = temp_dir  # сохраним путь для удаления
    return driver


def close_chrome_driver(driver):
    temp_dir = getattr(driver, "_temp_dir", None)
    try:
        driver.quit()
    except Exception:
        pass
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def human_sleep(a=1.2, b=3.5):
    time.sleep(random.uniform(a, b))


def load_image(image_link):
    """
    Load an image from a given link or file path.

    Args:
        image_link (str or np.ndarray): The image source (URL, file path, or numpy array).

    Returns:
        PIL.Image.Image or None: The loaded image, or None if loading fails.
    """
    if type(image_link) == str:
        if image_link.startswith("http"):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
                }
                response = requests.get(image_link, headers=headers)
                img = Image.open(BytesIO(response.content))
            except Exception as e:
                print(image_link)
                print(e)
                return None
        else:
            img = Image.open(image_link)
    elif type(image_link) == np.ndarray:
        img = Image.fromarray(image_link)
    else:
        raise Exception("Unknown image type")

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def encode_image(img):
    """
    Encode and normalize an image for model input.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        np.ndarray: The encoded and normalized image array.
    """
    img = img.resize(cfg.image_size)
    img = np.array(img)
    img = img.astype("float32")

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-7
    img = (img + epsilon) / (255.0 + epsilon)

    # Clip values to ensure they're in the valid range [0, 1]
    img = np.clip(img, 0, 1)

    return img


def load_data(image_link):
    """
    Load and preprocess an image from a given link.

    Args:
        image_link (str): The URL or file path of the image.

    Returns:
        tf.Tensor or None: The preprocessed image tensor, or None if loading fails.
    """
    img = load_image(image_link)
    if img is None:
        return None
    img = encode_image(img)

    # convert data to tf.tensor
    img = tf.convert_to_tensor(img)
    return img


class Processor(metaclass=RuntimeMeta):
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size
        self.session = requests.Session()
        self.user_agents = self.generate_similar_user_agents()
        self.headers_list = self.generate_headers_list()
        self.proxies = []

    def generate_similar_user_agents(self):
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        ]

    def generate_headers_list(self):
        base_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.goofish.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        headers_list = []
        for user_agent in self.user_agents:
            headers = base_headers.copy()
            headers["User-Agent"] = user_agent
            headers_list.append(headers)
        return headers_list

    def collect_product_links_selenium(self, max_steps=5, max_links=100):
        """
        Собирает ссылки на карточки товаров с помощью Selenium, эмулируя клики по пагинации.
        """
        driver = create_chrome_driver(
            user_agent=random.choice(self.user_agents), debug_port=9222
        )
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
            """
            },
        )

        driver.get("https://www.goofish.com")
        human_sleep(3, 7)

        cookies = [
            {
                "name": "_m_h5_tk",
                "value": "f7f7b405844996549b89e7d446ab17c7_1755010189196",
            },
            {"name": "_m_h5_tk_enc", "value": "48b7325338ec034774238119500289de"},
            {"name": "atpsida", "value": "479518a88d5ec1c53e46658a_1755000829_1"},
            {"name": "cna", "value": "B6kbIbUefW4CAZ611+tfYR/D"},
            {"name": "cna", "value": "B6kbITpMRmkCAZ611+t7b+sL"},
            {"name": "cookie2", "value": "1626f235bbcdd7170bef61fe4eb9101e"},
            {"name": "mtop_partitioned_detect", "value": "1"},
            {"name": "sca", "value": "a2d8698f"},
            {"name": "t", "value": "40000c4303f6358b27bd11c644531188"},
            {
                "name": "tfstk",
                "value": "gBJIaV02UUpackOAOToaho_OyXB5Fck4FusJm3eU29BLyzKA7e7FL0D5yebwLw-FpkAMuneeLXXzF9Xlequq3xoHxTX8Ukfatwx9jgnNvwnSGJBlequa_8CnhTYKCwENwhn14gy8wUCpXOIVWyUpywCTWisceaLJyNIOqgz8JJFpXAQG2TQJyTn6XNj5eaLRectO7TPO2b_eAmOw5Gurraxd58eJpX1ClFVze8p1A69JvNZUYdsCOZC-0PfkhnKpng9oD7_5Ag7L3ZOz37Z1i8s1uci_Z7XRFMDtoG_S86IGAmmsfyUh9Gj1Lci_ZrCdjGrifcaCw",
            },
            {"name": "xlly_s", "value": "1"},
        ]

        for cookie in cookies:
            driver.add_cookie(cookie)

        collected = []
        driver.get(cfg.mainpage_url_goofish)
        human_sleep(2.5, 5.5)

        try:
            wait = WebDriverWait(driver, 15)
            close_btn = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//img[contains(@src, "TB1QZN.CYj1gK0jSZFuXXcrHpXa")]')
                )
            )
            ActionChains(driver).move_to_element(close_btn).pause(
                random.uniform(0.2, 0.7)
            ).click().perform()
            human_sleep(1, 2)
            print("Попап закрыт")
        except Exception as e:
            print("Попап не найден или уже закрыт:", e)

        # Кликаем по кнопке поиска
        try:
            wait = WebDriverWait(driver, 20)
            search_btn = wait.until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        '//button[contains(@class, "search-icon--") and @type="submit"]',
                    )
                )
            )
            ActionChains(driver).move_to_element(search_btn).pause(
                random.uniform(0.3, 1.2)
            ).click().perform()
            human_sleep(2.5, 5.5)
            print("Поиск выполнен")
        except Exception as e:
            print("Кнопка поиска не найдена или не нажата:", e)

        for page in range(1, max_steps + 1):
            # Случайный скролл вверх/вниз
            driver.execute_script(f"window.scrollTo(0, {random.randint(0, 500)});")
            human_sleep(0.7, 1.7)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            product_links = soup.find_all("a", class_="feeds-item-wrap--rGdH_KoF")
            for link_tag in product_links:
                href = link_tag.get("href")
                img_tag = link_tag.find("img", class_="feeds-image--TDRC4fV1")
                img_src = img_tag.get("src") if img_tag else None
                if href and href.startswith("/"):
                    href = "https://www.goofish.com" + href
                collected.append((img_src, href))
                if len(collected) >= max_links:
                    close_chrome_driver(driver)
                    return collected

            if page == max_steps:
                break
            # Скроллим вниз перед переходом
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            human_sleep(1.5, 3.5)
            try:
                wait = WebDriverWait(driver, 15)
                next_arrow = wait.until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            '(//button[starts-with(@class, "search-pagination-arrow-container--")])[2]',
                        )
                    )
                )
                ActionChains(driver).move_to_element(next_arrow).pause(
                    random.uniform(0.3, 1.2)
                ).click().perform()
                human_sleep(2.5, 5.5)
            except Exception as e:
                print(f"Не удалось перейти на страницу {page+1} по стрелке: {e}")
                break

        close_chrome_driver(driver)
        return collected

    def parse_big_images_from_slider_selenium(self, page_url):
        """
        Собирает все большие картинки из слайдера карточки товара goofish через Selenium.
        """
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(page_url)
        time.sleep(3)

        image_links = set()

        carousel_imgs = driver.find_elements(
            By.XPATH,
            '//div[starts-with(@class, "carouselItem--")]//img[contains(@style, "width: 100%")]',
        )
        for img in carousel_imgs:
            src = img.get_attribute("src")
            if src and "alicdn.com" in src:
                if src.startswith("//"):
                    src = "https:" + src
                image_links.add(src)

        try:
            active_img = driver.find_element(
                By.XPATH,
                '//img[contains(@style, "width: 100%") and contains(@style, "height: 100%") and not(ancestor::div[starts-with(@class, "carouselItem--")])]',
            )
            src = active_img.get_attribute("src")
            if src and "alicdn.com" in src:
                if src.startswith("//"):
                    src = "https:" + src
                image_links.add(src)
        except Exception:
            pass

        close_chrome_driver(driver)
        return list(image_links)

    def load_product_info(self, page_url):
        """
        Парсит цену товара с карточки goofish.
        """
        try:
            headers = random.choice(self.headers_list)
            response = self.session.get(page_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            # Ищем любой тег с классом, содержащим "price--"
            price_tag = soup.find(
                lambda tag: tag.has_attr("class")
                and any("price--" in c for c in tag["class"])
            )
            price = price_tag.text.strip() if price_tag else "N/A"
            return {"price": price}
        except Exception as e:
            logging.warning(f"Could not load product info for {page_url}: {e}")
            return {"price": "N/A"}

    def load_product_info_selenium(self, page_url):
        driver = create_chrome_driver(
            user_agent=random.choice(self.user_agents), debug_port=9223
        )
        driver.get(page_url)
        time.sleep(2)
        try:
            price_elem = driver.find_element(
                By.XPATH, '//*[contains(@class, "price--")]'
            )
            price = price_elem.text.strip()
        except Exception:
            price = "N/A"
        close_chrome_driver(driver)
        return {"price": price}

    def process_encoded_data(self, encoded_data, result):
        """
        Обрабатывает закодированные данные товара и добавляет их в результат.

        Args:
            encoded_data (dict): Закодированные данные о товаре.
            result (dict): Результирующий словарь для записи в Excel.

        Returns:
            None
        """
        for k, v in encoded_data.items():
            if k == "incorrect_image_links":
                result[k].append(", ".join(v) if isinstance(v, list) else str(v))
            else:
                result[k].append(v)

from picker_model import TargetModel 

import argparse
import re

import json 
import os 
from IPython.display import clear_output


def collect_links(t, first_page_link, max_pages=3, max_links=90, verbose=0) -> list:
    products_links = list()
    for i in range(max_pages):
        page_num = i + 1
        main_link = first_page_link.replace("b=1", f"b={1 + (page_num - 1) * 100}")

        # Добавьте этот блок для отладки:
        # print(f"===> Парсим страницу: {main_link}")
        # for img, link in t.processor.get_page_content(main_link):
        #     print("RAW LINK:", link)


        auction_pattern = re.compile(r"^https://auctions\.yahoo\.co\.jp/jp/auction/[a-zA-Z0-9]+$")

        pages = [
            link
            for _, link in t.processor.get_page_content(main_link)
            if auction_pattern.match(link)
        ]

        if verbose:
            print("\n".join(pages))

        products_links.extend(pages)

        if len(products_links) >= max_links:
            break

    return products_links[:max_links]


# def encode_images(t, page_link):
#     image_links = t.processor.parse_images_from_page(page_link)
#     image_links = list(set(image_links))

#     target_link = t.do_inference_minimodel(image_links)

#     print(target_link)
#     clear_output(wait=True)
#     return {
#       **{k: 0 for k in [link for link in image_links if link != target_link]},
#       **{target_link: 1},
#   }


# def map_fn(t, target_folder, page_link):

#   if not os.path.exists(target_folder):
#     os.mkdir(target_folder)

#   if os.path.exists(f'{target_folder}/{page_link.split("/")[-1]}.json'):
#     return

#   predicted_data = encode_images(t, page_link)

#   # save predicted_data with name page.split('/')[-1] by json
#   # if it already exists dont save

#   with open(f'{target_folder}/{page_link.split("/")[-1]}.json', 'w') as f:
#     json.dump(predicted_data, f)


# def main(main_page_link, target_folder_name) -> None :
#   t = TargetModel()

#   products_links = collect_links(t, main_page_link)

#   # remove duplicates
#   products_links = list(set(products_links))

#   for i, page_link in enumerate(products_links):
#     print(f'page {i+1}/{len(products_links)}')
#     map_fn(t, target_folder_name, page_link)

# def parse_args():
#     """
#     Main usage Example:

#         python script.py --page-link "https://example.com/page-link" --folder-name "target_folder"

#     """

#     parser = argparse.ArgumentParser(description="Arguments for running the image encoding and link collection script")

#     parser.add_argument('--page-link', type=str, required=True, help="The main page link to start collecting product links from")
#     parser.add_argument('--folder-name', type=str, required=True, help="The target folder name where the JSON files will be saved")

#     args = parser.parse_args()

#     return args.page_link, args.folder_name

# if __name__ == '__main__':
#   page_link, folder_name = parse_args()
#   main(page_link, folder_name)

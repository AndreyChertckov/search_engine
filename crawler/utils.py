import logging, time

import requests

from config import SEARCHER_URL


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Crawler")


def update_searcher(url, text, title, retries=5):
    if retries <= 0:
        return 

    try:
        r = requests.post(SEARCHER_URL, 
                json={"url": url, "text": text, "title": title})
    except requests.ConnectionError:
        logger.error(f"Cannot connect to searcher {SEARCHER_URL}")
        time.sleep(10)
        return update_searcher (url, text, title, retries-1)

    if r.status_code != 200:
        logger.error(f"Searcher returns non 200 code. {r.status_code} : {r.text}")
        return

    logger.info(f"Searcher returns 200 code. {r.status_code} : {r.text}")


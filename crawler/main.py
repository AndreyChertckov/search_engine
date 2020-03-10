import time

from crawler import Crawler
from utils import update_searcher


def main():
    crawler = Crawler()

    for c in crawler.crawl_generator("https://en.wikipedia.org/wiki/Shrek", 2):
        if c.doc.url[-4:] in ('.pdf', '.mp3', '.avi', '.mp4', '.txt'):
            continue
        url = c.doc.url
        text = c.doc.text
        title = c.doc.title
        update_searcher(url, text, title)
        time.sleep(2)


if __name__ == "__main__":
    main()

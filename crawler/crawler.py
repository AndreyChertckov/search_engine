import re
from queue import Queue
from urllib.parse import quote
import urllib.parse
from collections import Counter

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment


class Document:
    
    def __init__(self, url):
        self.url = url
        self.content = b''
        
    def get(self):
        if not self.load():
            if not self.download():
                raise FileNotFoundError(self.url)
            else:
                self.persist()
    
    def download(self):
        try:
            r = requests.get(quote(self.url, safe=':/'), stream=True)
        except requests.ConnectionError:
            return False
        if r.status_code == 200:
            self.content = b''
            for chunk in r:
                self.content += chunk
            return True
        else:
            return False
    
    def persist(self):
        with open(self.url.replace('/', '.'), 'wb') as f:
            f.write(self.content)
            
    def load(self):
        try:
            with open(self.url.replace('/', '.'), 'rb') as f:
                self.content = f.read()
            return True
        except FileNotFoundError:
            return False


class HtmlDocument(Document):
    
    def parse(self):
        self.bs = BeautifulSoup(self.content.decode('utf-8'))
        self.anchors = self.__find_all_anchors()
        self.images = self.__find_all_images()
        self.text = self.text_from_html()
        self.title = self.get_title()
        
    def __find_all_anchors(self):
        anchors = []
        for link in self.bs.find_all('a'):
            href = link.get("href")
            if not href:
                continue
            if not href.startswith("http"):
                href = urllib.parse.urljoin(self.url, href)
            anchors.append((link.string, href))
        return anchors
    
    def __find_all_images(self):
        images = []
        for img in self.bs.find_all('img'):
            src: str = img.get("src")
            if not src:
                continue
            if not src.startswith("http"):
                src = urllib.parse.urljoin(self.url, src)
            images.append(src)
        return images
    
    @staticmethod
    def __tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def get_title(self):
        return self.bs.title.string
    
    def text_from_html(self):
        texts = self.bs.findAll(text=True)
        visible_texts = filter(self.__tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)


class HtmlDocumentTextData:
    
    def __init__(self, url):
        self.doc = HtmlDocument(url)
        self.doc.get()
        self.doc.parse()
    
    def get_sentences(self):
        return re.findall(r"\w+", self.doc.text.lower())
    
    def get_word_stats(self):
        return Counter(self.get_sentences())


class Crawler:
    
    def crawl_generator(self, source, depth=1):
        visited = []
        q = list()
        q.append(source)
        next_level_queue = []
        for i in range(depth):
            while len(q):
                s = q.pop()
                visited.append(s)
                try:
                    html_document_text_data = HtmlDocumentTextData(s)
                except Exception as e:
                    print(e)
                    continue                
                anchors = html_document_text_data.doc.anchors
                for _,a in anchors:
                    if a not in visited and a not in next_level_queue:
                        next_level_queue.append(a)
                yield html_document_text_data
            q = next_level_queue[:]
            next_level_queue = []
        

if __name__ == "__main__":
    crawler = Crawler()
    counter = Counter()

    for c in crawler.crawl_generator("https://university.innopolis.ru/en/", 2):
        print(c.doc.url)
        if c.doc.url[-4:] in ('.pdf', '.mp3', '.avi', '.mp4', '.txt'):
            print("Skipping", c.doc.url)
            continue
        counter.update(c.get_word_stats())
        print(len(counter), "distinct word(s) so far")
        
    print("Done")

    print(counter.most_common(20))
    assert [x for x in counter.most_common(20) if x[0] == 'innopolis'], 'innopolis sould be among most common'


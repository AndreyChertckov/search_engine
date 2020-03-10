import time
import pickle
import logging

from pymongo import MongoClient
import redis
import requests

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("worker")

def check_url(url):
    try:
        r = requests.get(url)
    except requests.ConnectionError:
        return False
    return r.status_code == 200


def save_aux_to_mongo(mongo_client, redis_client):
    while True:
        time.sleep(30)
        keys = redis_client.keys("index:*")
        keys = [k.decode("utf-8") for k in keys]
        deleted = []
        checked = {} 
        for key in keys:
            word = key[6:]
            docs = redis_client.smembers(key)
            docs_filtered = []
            for doc in docs:
                title, url = pickle.loads(doc)

                if url not in checked:
                    checked[url] = check_url(url)
                
                if checked[url]:
                    docs_filtered.append((title, url))
                else:
                    deleted.append(url)

            index_word = mongo_client.index.main.find_one({"word": word})
            if index_word is None:
                mongo_client.index.main.insert_one({"word": word, "docs": docs_filtered})
            else:
                mongo_client.index.main.update_one({"_id": index_word["_id"]}, {"$set": {"docs": docs_filtered}})
            redis_client.delete(key)
        logger.info(f"Saved: {len(keys)}")
        logger.info(f"Deleted: {len(deleted)}")


def main():
    mongo_client = MongoClient("mongodb://search_user:qdrwbj123@mongo/?authSource=index")
    redis_client = redis.Redis(host="redis", db=0)
    save_aux_to_mongo(mongo_client, redis_client)


if __name__ == "__main__":
    main()

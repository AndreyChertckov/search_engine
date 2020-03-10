from collections import defaultdict

from flask import Flask, request, render_template
from pymongo import MongoClient
import pymongo
import redis

from search import *


app = Flask(__name__, template_folder="./templates/")
mongo_client = MongoClient("mongodb://search_user:qdrwbj123@mongo/?authSource=index")
index_db = mongo_client.index
redis_client = redis.Redis(host="redis", db=0)


@app.route("/query", methods=["GET"])
def searche():
    q = request.args.get("q", None)
    if q is None:
        return render_template("index_template.html", results=[])

    search_result, titles = search(index_db, redis_client, q)
    results = [{"url": url, "title": titles[url]} for url in search_result]
    if results == []:
        return render_template("index_template.html", error="Nothing found")
    return render_template("index_template.html", results=results)


@app.route("/update", methods=["POST"])
def update():
    global index, soundex, tree

    data = request.json
    if "url" not in data or "text" not in data or "title" not in data:
        return "error", 400
    update_indexes(index_db.main, redis_client, data["url"], data["title"], data["text"])
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

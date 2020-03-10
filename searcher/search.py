import re
import string
import unicodedata
import pickle
from collections import defaultdict, deque

import numpy as np
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def normalize(text):
    text = text.replace("}", "}}")
    text = text.replace("{", "{{")
    numbers = [int(s) for s in text.split() if s.isdigit()]
    numbers_words = map(num2words, numbers)
    without_number = re.sub(r" \d+ ", " {} ", text)
    without_number = re.sub(r"\d+", "", without_number)
    with_numbers = without_number.format(*numbers_words)
    stars = re.sub(r"\*+", "*", with_numbers)
    lowercased = stars.lower()
    punct = string.punctuation.replace("*", "")
    without_punct = lowercased.translate(str.maketrans('', '', punct))
    without_whitespaces = without_punct.replace(string.whitespace, ' ')
    without_accents = unicodedata.normalize('NFKD', without_whitespaces).encode('ASCII', 'ignore').decode('utf-8')
    return without_accents

    
def num2words(num):
    under_20 = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Eleven','Twelve','Thirteen','Fourteen','Fifteen','Sixteen','Seventeen','Eighteen','Nineteen']
    tens = ['Twenty','Thirty','Forty','Fifty','Sixty','Seventy','Eighty','Ninety']
    above_100 = {100: 'Hundred',1000:'Thousand', 1000000:'Million', 1000000000:'Billion'}

    if num < 20:
        return under_20[num]

    if num < 100:
        return tens[(int)(num/10)-2] + ('' if num%10==0 else ' ' + under_20[num%10])

    # find the appropriate pivot - 'Million' in 3,603,550, or 'Thousand' in 603,550
    pivot = max([key for key in above_100.keys() if key <= num])

    return num2words((int)(num/pivot)) + ' ' + above_100[pivot] + ('' if num%pivot==0 else ' ' + num2words(num%pivot))


def tokenize(text):
    return word_tokenize(text)


def lemmatization(tokens):
    return list(map(lemmatizer.lemmatize, tokens))


def remove_stop_word(tokens):
    return list(filter(lambda x: x not in stop_words, tokens))


def preprocess(text):
    normalized = normalize(text)
    tokenized = tokenize(normalized)
    lemmatized = lemmatization(tokenized)
    without_stop_words = remove_stop_word(lemmatized)
    return without_stop_words


def to_soundex(word):
    rules = {
        "a": 0,
        "e": 0,
        "o": 0,
        "i": 0,
        "u": 0,
        "h": 0,
        "w": 0,
        "y": 0,
        "b": 1,
        "f": 1,
        "p": 1,
        "v": 1,
        "c": 2,
        "g": 2,
        "j": 2,
        "k": 2,
        "q": 2,
        "s": 2,
        "x": 2,
        "z": 2,
        "d": 3,
        "t": 3,
        "l": 4,
        "m": 5,
        "n": 5,
        "r": 6
    }
    result = word[0]
    previous = -1
    for l in word[1:]:
        if l not in rules:
            continue
        n = rules[l]
        if previous == n:
            continue
        previous = n
        if n != 0:
            result += str(n)
    result += "0000"
    return result[:4]


def find_word(index_collection, redis_client, word):
    docs = redis_client.smembers(f"index:{word}")
    if len(docs) == 0:
        result = index_collection.find_one({"word": word})
        docs = result["docs"]
        docs_to_save = list(map(pickle.dumps, docs))
        redis_client.sadd(f"index:{word}", *docs_to_save)
    else:
        docs = list(map(pickle.loads, docs))
    return docs


def update_word(index_collection, redis_client, idx, title, word):
    if redis_client.exists(f"index:{word}"):
        redis_client.sadd(f"index:{word}", pickle.dumps((title, idx)))
    else:
        index_word = index_collection.find_one({"word": word})
        if index_word is None:
            redis_client.sadd(f"index:{word}", pickle.dumps((title, idx)))
        else:
            docs = index_word["docs"]
            docs.append((title, idx))
            docs = list(map(pickle.dumps, docs))
            redis_client.sadd(f"index:{word}", *docs)


def update_indexes(index_collection, redis_client, idx, title, text):
    words = set(preprocess(text))
    for word in words:
        update_word(index_collection, redis_client, idx, title, word)
        s = to_soundex(word)
        redis_client.sadd(f"soundex:{s}", word)
    tree = get_permuted_tree(redis_client)
    tree = update_permuted_tree(tree, text)
    save_permuted_tree(redis_client, tree)


def soundex_search(redis_client, query):
    s = to_soundex(query)
    result = redis_client.smembers(f"soundex:{s}")
    result = [b.decode("utf-8") for b in list(set(result))]
    return list(result)


def traverse(tree, prefix):
    result = []
    if tree == {} or tree["end"]:
        result = [prefix]
    for k in tree:
        if k != "end":
            result.extend(traverse(tree[k], prefix + k))
    return result


def search_tree(tree, query):
    left, right = query.split("*", maxsplit=1)
    processed_query = right + "$" + left + "*"
    cur = tree
    prefix = ""
    result = []
    for letter in processed_query:
        if letter == "*":
            result.extend(traverse(cur, prefix))
        else:
            if letter in cur:
                prefix += letter
                cur = cur[letter]
            else:
                print(letter)
                raise Exception
    processed_output = []
    for res in result:
        left, right = res.split("$", maxsplit=1)
        processed_output.append(right + left)
    return processed_output


def get_permuted_tree(redis_client):
    bin_tree = redis_client.get("permuted_tree")
    if bin_tree is None:
        tree = get_empty_permuted_tree()
    else:
        tree = pickle.loads(bin_tree)
    return tree


def save_permuted_tree(redis_client, tree):
    bin_tree = pickle.dumps(tree)
    redis_client.set("permuted_tree", bin_tree)


def get_empty_permuted_tree():
    return {"end": False}


def build_permuted_tree(collection):
    tree = {"end": False}
    for text in tqdm(collection):
        tree = update_permuted_tree(tree, text)

    return tree


def update_permuted_tree(tree, text):
    for word in preprocess(text):
        word_for_permutation = "$" + word
        d = deque(word_for_permutation)
        for i in range(len(word_for_permutation)):
            cur_word = ''.join(list(d))
            cur_node = tree
            for letter in cur_word:
                if letter not in cur_node:
                    cur_node[letter] = {"end": False}
                cur_node = cur_node[letter]
            cur_node["end"] = True
            d.rotate(1)
    return tree


def levenshtein(word, query):
    size_x = len(word) + 1
    size_y = len(query) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if word[x-1] == query[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )   
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def find_closest_levenshtein(indexes, query, max_distance=2):
    
    results = []
    for word in indexes:
        d = levenshtein(word, query)
        results.append((d, word))
    result_words = []
    for res in sorted(results):
        if res[0] < max_distance:
            result_words.append(res[1])
    return result_words


def search(index_db, redis_client, query):
    proccessed_query = preprocess(query)
    result_query = []
    for q in proccessed_query:
        if "*" in q:
            tree = get_permuted_tree(redis_client)
            result_query += [search_tree(tree, q)]
        elif index_db.main.count_documents({"word": q}):
            result_query += [[q]]
        else:
            result_query += [find_closest_levenshtein(soundex_search(redis_client, q), q)]
    results = []
    titles = defaultdict(str)
    for sub_q in result_query:
        sub_q_result = []
        for q in sub_q:
            docs = find_word(index_db.main, redis_client, q)
            q_result = []
            for doc in docs:
                title, url = doc
                q_result.append(url)
                titles[url] = title
            sub_q_result.append(q_result)
        results.append(sub_q_result)
    relevant_sub_indx = [list(set().union(*sub_ind)) for sub_ind in results]
    if not len(relevant_sub_indx):
        return [], titles
    relevant_indx = list(set(relevant_sub_indx[0]).intersection(*relevant_sub_indx[1:]))
    return relevant_indx, titles


if __name__ == "__main__":
    from pymongo import MongoClient
    import pymongo
    import redis
    mongo_client = MongoClient("mongodb://search_user:qdrwbj123@mongo/?authSource=index")
    index_db = mongo_client.index
    redis_client = redis.Redis(host="redis", db=0)
    tree = {"end": True}
    result = search(index_db, redis_client, tree, "shrak")
    print(result)


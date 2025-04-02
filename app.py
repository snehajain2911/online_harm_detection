from flask import Flask, jsonify, render_template, request
import requests
import json
import cv2
import pytesseract
from PIL import Image
import os
import praw
import tweepy
from googleapiclient.discovery import build
from flask_cors import CORS
import logging
import time
import pickle
import string
import nltk
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pymongo import MongoClient
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from werkzeug.utils import secure_filename

load_dotenv()
nltk.download('stopwords')

app = Flask(__name__, template_folder="templates")
CORS(app)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Lazy load MongoDB connection
def get_mongo_collection():
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client[os.getenv("DB_NAME")]
    return db[os.getenv("COLLECTION_NAME")]

# Lazy load Google Perspective API Key
def get_google_api_key():
    return os.getenv("GOOGLE_API_KEY")

# Lazy load models
def get_cyberbullying_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def get_misinformation_model():
    return (AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased"),
            AutoTokenizer.from_pretrained("distilbert-base-uncased"))

# Lazy load APIs
def get_reddit_client():
    return praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),
                        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                        user_agent=os.getenv("REDDIT_USER_AGENT"))

def get_twitter_client():
    return tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))

def get_youtube_client():
    return build("youtube", "v3", developerKey=os.getenv("GOOGLE_API_KEY"))

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

# Cyberbullying Detection
def predict_cyberbullying(text):
    model, vectorizer = get_cyberbullying_model()
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized_text)
    return "Cyberbullying Detected" if prediction[0] == 1 else "No Cyberbullying Detected"

# Misinformation Detection
def verify_information(claim):
    model, tokenizer = get_misinformation_model()
    inputs = tokenizer(claim, return_tensors="pt")
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
    return "True" if scores[1] > scores[0] else "False"

# Hate Speech Detection
def detect_hate_speech(text):
    api_key = get_google_api_key()
    data = {"comment": {"text": text}, "languages": ["en"], "requestedAttributes": {"TOXICITY": {}, "INSULT": {}, "THREAT": {}}}
    response = requests.post("https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze", params={"key": api_key}, json=data)
    return response.json()

# Store flagged content
def store_flagged_content(platform, content, url, category, score):
    collection = get_mongo_collection()
    collection.insert_one({"platform": platform, "content": content, "url": url, "category": category, "score": score})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/reddit_feed")
def reddit_feed():
    reddit = get_reddit_client()
    posts = []
    for submission in reddit.subreddit("all").new(limit=10):
        text_analysis = detect_hate_speech(submission.title)
        cyberbullying_prediction = predict_cyberbullying(submission.title)
        misinformation = verify_information(submission.title)
        posts.append({"title": submission.title, "url": submission.url, "hate_speech": text_analysis, "cyberbullying": cyberbullying_prediction, "misinformation": misinformation})
    return jsonify(posts)

@app.route("/twitter_feed")
def twitter_feed():
    client = get_twitter_client()
    tweets = []
    response = client.search_recent_tweets(query="technology -is:retweet lang:en", max_results=10, tweet_fields=["created_at"])
    if response.data:
        for tweet in response.data:
            text_analysis = detect_hate_speech(tweet.text)
            cyberbullying_prediction = predict_cyberbullying(tweet.text)
            misinformation = verify_information(tweet.text)
            tweets.append({"text": tweet.text, "created_at": str(tweet.created_at), "hate_speech": text_analysis, "cyberbullying": cyberbullying_prediction, "misinformation": misinformation})
    return jsonify(tweets)

@app.route("/youtube_feed")
def youtube_feed():
    youtube = get_youtube_client()
    videos = []
    request = youtube.videos().list(part="snippet", chart="mostPopular", regionCode="US", maxResults=10)
    response = request.execute()
    for item in response["items"]:
        title = item["snippet"]["title"]
        text_analysis = detect_hate_speech(title)
        cyberbullying_prediction = predict_cyberbullying(title)
        misinformation = verify_information(title)
        videos.append({"title": title, "channel": item["snippet"]["channelTitle"], "hate_speech": text_analysis, "cyberbullying": cyberbullying_prediction, "misinformation": misinformation})
    return jsonify(videos)

@app.route("/reporting")
def reporting():
    flagged_content = list(get_mongo_collection().find())
    return render_template("reporting.html", flagged_content=flagged_content)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

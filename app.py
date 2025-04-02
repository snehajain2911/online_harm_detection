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

load_dotenv()  # Load .env file

nltk.download('stopwords')


# ✅ Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)
# ✅ MongoDB Connection 
# ✅ MongoDB Connection
 # Default if not set
MONGO_URL = os.getenv("MONGO_URL") 
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# ✅ Connect to MongoDB
client = MongoClient(MONGO_URL)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
#db = client[os.getenv("DB_NAME", "online_harm_detection")]
#collection = db[os.getenv("COLLECTION_NAME", "flagged_content")]


# ✅ Set Tesseract OCR path (Required for Windows)

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# ✅ Google Perspective API Setup
API_KEY = os.getenv("GOOGLE_API_KEY")
URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# ✅ Reddit API Setup
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
subreddit = reddit.subreddit("all")

# ✅ Twitter API Setup
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
# ✅ YouTube API Setup
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# ✅ Load Cyberbullying Model and Vectorizer
with open("model.pkl", "rb") as model_file:
    cyberbullying_model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ✅ Load Misinformation Detection Model

misinfo_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
misinfo_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")



# ✅ Text Cleaning for Cyberbullying Detection
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

# ✅ Cyberbullying Detection
def predict_cyberbullying(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = cyberbullying_model.predict(vectorized_text)
    return "Cyberbullying Detected" if prediction[0] == 1 else "No Cyberbullying Detected"

# ✅ Misinformation Detection
def verify_information(claim):
    inputs = misinfo_tokenizer(claim, return_tensors="pt")
    outputs = misinfo_model(**inputs)
    scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
    result = "True" if scores[1] > scores[0] else "False"
    return result

# 🚨 Hate Speech Detection
def detect_hate_speech(text):
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}, "INSULT": {}, "THREAT": {}}
    }
    response = requests.post(URL, params={"key": API_KEY}, json=data)
    result = response.json()

    if "error" in result:
        return {"error": result['error']['message']}

    toxicity = round(result["attributeScores"]["TOXICITY"]["summaryScore"]["value"] * 100, 2)
    insult = round(result["attributeScores"]["INSULT"]["summaryScore"]["value"] * 100, 2)
    threat = round(result["attributeScores"]["THREAT"]["summaryScore"]["value"] * 100, 2)

    def categorize(score):
        if score <= 5:
            return "Safe"
        elif score <= 15:
            return "Low Risk"
        elif score <= 30:
            return "Mild Risk"
        else:
            return "Risky"

    return {
        "toxicity": {"score": f"{toxicity}%", "category": categorize(toxicity)},
        "insult": {"score": f"{insult}%", "category": categorize(insult)},
        "threat": {"score": f"{threat}%", "category": categorize(threat)}
    }

# 📸 Image Text Extraction
def extract_text(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "❌ Error: Could not read image."
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(Image.fromarray(gray)).strip()
    return text if text else "No text detected."


# 🚩 Store Flagged Content in MongoDB
def store_flagged_content(platform, content, url, category, score):
    """Store flagged content in MongoDB."""
    flagged_data = {
        "platform": platform,
        "content": content,
        "url": url,
        "category": category,
        "score": score
    }
    collection.insert_one(flagged_data)
    

# ✅ Home Route
@app.route("/")
def home():
    return render_template("index.html")
# ✅ Reddit Feed
@app.route("/reddit_feed")
def reddit_feed():
    posts = []
    for submission in subreddit.new(limit=10):
        text_analysis = detect_hate_speech(submission.title)
        cyberbullying_prediction = predict_cyberbullying(submission.title)
        misinformation = verify_information(submission.title)

        # Check for risky content and store it in MongoDB
        if "Risky" in [
            text_analysis['toxicity']['category'], 
            text_analysis['insult']['category'], 
            text_analysis['threat']['category']
        ]:
            store_flagged_content("Reddit", submission.title, submission.url, "Risky", text_analysis['toxicity']['score'])

        posts.append({
            "title": submission.title,
            "url": submission.url,
            "subreddit": str(submission.subreddit),
            "hate_speech": text_analysis,
            "cyberbullying": cyberbullying_prediction,
            "misinformation": misinformation
        })
    return jsonify(posts)

# ✅ Twitter Feed
@app.route("/twitter_feed")
def twitter_feed():
    tweets = []
    query = "technology -is:retweet lang:en"
    response = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["created_at"])
    if response.data:
        for tweet in response.data:
            text_analysis = detect_hate_speech(tweet.text)
            cyberbullying_prediction = predict_cyberbullying(tweet.text)
            misinformation = verify_information(tweet.text)

            # Check for risky content and store it in MongoDB
            if "Risky" in [
                text_analysis['toxicity']['category'], 
                text_analysis['insult']['category'], 
                text_analysis['threat']['category']
            ]:
                store_flagged_content("Twitter", tweet.text, "N/A", "Risky", text_analysis['toxicity']['score'])

            tweets.append({
                "text": tweet.text,
                "created_at": str(tweet.created_at),
                "hate_speech": text_analysis,
                "cyberbullying": cyberbullying_prediction,
                "misinformation": misinformation
            })
    return jsonify(tweets)

# ✅ YouTube Feed
@app.route("/youtube_feed")
def youtube_feed():
    videos = []
    request = youtube.videos().list(part="snippet", chart="mostPopular", regionCode="US", maxResults=10)
    response = request.execute()
    for item in response["items"]:
        title = item["snippet"]["title"]
        text_analysis = detect_hate_speech(title)
        cyberbullying_prediction = predict_cyberbullying(title)
        misinformation = verify_information(title)

        # Check for risky content and store it in MongoDB
        if "Risky" in [
            text_analysis['toxicity']['category'], 
            text_analysis['insult']['category'], 
            text_analysis['threat']['category']
        ]:
            store_flagged_content("YouTube", title, "N/A", "Risky", text_analysis['toxicity']['score'])

        videos.append({
            "title": title,
            "channel": item["snippet"]["channelTitle"],
            "hate_speech": text_analysis,
            "cyberbullying": cyberbullying_prediction,
            "misinformation": misinformation
        })
    return jsonify(videos)
# YouTube Video Analysis Route
@app.route("/youtube_video_analysis", methods=["GET"])
def youtube_video_analysis():
    video_id = request.args.get("video_id")
    if not video_id:
        return jsonify({"error": "Missing video ID"}), 400
    
    comments = get_video_comments(video_id)
    transcript = get_video_transcript(video_id)
    analyzed_comments = [
        {
            "comment": comment,
            "hate_speech": detect_hate_speech(comment),
            "cyberbullying": predict_cyberbullying(comment),
            "misinformation": verify_information(comment)
        }
        for comment in comments
    ]
    return jsonify({"comments": analyzed_comments, "transcript": transcript})

# Fetch YouTube Comments
def get_video_comments(video_id, max_results=50):
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText")
    response = request.execute()
    for item in response.get("items", []):
        comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    return comments

# Fetch YouTube Transcript
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except:
        return "Transcript not available."

# ✅ Reporting Route to Display Flagged Content
@app.route("/reporting")
def reporting():
    # Fetch flagged content from MongoDB
    flagged_content = list(collection.find())
    return render_template("reporting.html", flagged_content=flagged_content)

# Ensure the upload directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 📸 Image Upload and Processing
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files["image"]
    caption = request.form.get("caption", "")
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(image.filename))
    image.save(image_path)

    extracted_text = extract_text(image_path)  # Function to extract text from image
    hate_speech_analysis = detect_hate_speech(extracted_text)
    cyberbullying_prediction = predict_cyberbullying(extracted_text)
    misinformation = verify_information(extracted_text)

    return jsonify({
        "extracted_text": extracted_text,
        "hate_speech": hate_speech_analysis,
        "cyberbullying": cyberbullying_prediction,
        "misinformation": misinformation
    })




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port)

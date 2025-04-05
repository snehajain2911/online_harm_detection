# Scaling Trust: AI-Powered Detection of Online Harms

## 📖 Description
The rapid growth of online platforms has led to an exponential increase in harmful content, including **hate speech, misinformation, and cyberbullying**. Traditional moderation methods struggle to keep up with the volume and complexity of these threats, leaving platforms vulnerable to abuse and users exposed to harm. This project leverages **AI-powered detection** to identify and flag harmful content in real time across multiple platforms.

---

## 📂 Project Structure
```
ONLINE_HARM_DETECTION/
│── templates/             # HTML templates for the web interface
│   ├── index.html         # Main page for user interaction
│   ├── reporting.html     # Reporting interface for flagged content
│── uploads/               # Directory for storing uploaded files
│── .env                   # Environment variables (API keys, configurations)
│── app.py                 # Main application script
│── model.pkl              # Trained machine learning model
│── online_harm.db         # SQLite database for storing reports and data
│── README.md              # Project documentation
│── requirements.txt       # Dependencies required to run the project
│── vectorizer.pkl         # Pre-trained vectorizer for text processing
```

---

## 📊 How It Works
1️⃣ **Fetches Data** → Collects **live data** from **Reddit, Twitter, YouTube**  
2️⃣ **Processes Content** → Applies **NLP & AI models** for analysis  
3️⃣ **Flags Risky Content** → Stores **harmful content** in **MongoDB Atlas**  
4️⃣ **Displays Dashboard** → Shows **real-time reports** in the **web UI**  

---

## ✨ Features
✅ **AI-powered detection** of **hate speech, misinformation, and cyberbullying**  
✅ **Real-time** content moderation across **Reddit, Twitter, and YouTube**  
✅ **Image text analysis** using **OCR (Tesseract)**  
✅ **Automated reporting** of **risky content**  
✅ **Manual Image Upload — upload images directly for harm analysis**
✅ **Flagged Content Dashboard — see content, source, platform, and risk score**
✅ **MongoDB Atlas Integration for scalable, cloud-based storage**
✅ **Customizable AI models for improved accuracy and adaptability**
---

## 🛠️ Technologies Used
| **Component**            | **Technology Used**                                         |
|--------------------------|------------------------------------------------------------|
| **Web Framework**        | Flask                                                      |
| **Frontend**             | HTML, CSS, JavaScript                                     |
| **Hate Speech Detection**| Google Perspective API                                    |
| **Cyberbullying Model**  | Trained ML model (Scikit-learn, NLP techniques)          |
| **Misinformation Model** | Transformers (Hugging Face - BERT/DeBERTa-based models)  |
| **Database**             | MongoDB atlas(for flagged content storage)                     |
| **OCR for Image Analysis** | Tesseract OCR                                           |
| **Data Sources**         | Reddit (PRAW), Twitter (Tweepy), YouTube API              |

---

## 🚀 Setup Instructions
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/online-harm-detection.git
cd online-harm-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys
Create a **`.env`** file and add your API keys:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
TWITTER_BEARER_TOKEN=your_bearer_token
YOUTUBE_API_KEY=your_api_key
PERSPECTIVE_API_KEY=your_api_key
```

---

## 📌 Deployment Options
You can deploy this project using **Google App Engine, Render, or PythonAnywhere**.

### 🔹 Google App Engine
- Follow [this guide](https://cloud.google.com/appengine/docs/standard/python3/quickstart) for deployment.

### 🔹 Render
#### Steps:
1. **Connect your GitHub repo**
2. **Choose Python environment**
3. **Set up `.env` variables**
4. **Start MongoDB Server (Required for Storing Flagged Content)**
    - If MongoDB is installed locally, start it using:
      ```sh
      mongod
      ```
    - If using MongoDB Compass (GUI version):
      - Open MongoDB Compass.
      - Connect to `mongodb://localhost:27017/`.
      - Ensure the `online_harm_detection` database is created automatically when the app runs.
5. **Run the Flask App**
    ```sh
    python app.py
    ```
    Access the web app at **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**.

---




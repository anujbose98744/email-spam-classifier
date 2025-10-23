📧 Email Spam Classifier

A Machine Learning web app that detects whether a given email or message is Spam or Not Spam using Natural Language Processing (NLP) techniques and a trained classifier.

🧩 Project Overview

The Email Spam Classifier helps identify spam messages automatically by analyzing text content.
It uses text preprocessing, feature extraction (TF-IDF), and ML algorithms like Naive Bayes or Logistic Regression to make accurate predictions.
The frontend is built using Streamlit, offering a clean, interactive interface.

🚀 Features

✅ Real-time spam prediction

🧠 Trained using NLP and ML techniques

🧹 Text preprocessing (tokenization, stopword removal, stemming)

📊 Dataset analysis and visualizations

🌐 User-friendly Streamlit web interface

💾 Model serialization using pickle

🧠 Tech Stack
Category	Tools
Programming Language	Python
Libraries / Frameworks	Scikit-learn, NLTK, Pandas, NumPy, Streamlit
Visualization	Matplotlib, Seaborn
Model Deployment	Streamlit / Heroku
⚙️ How It Works

Data Collection – Uses a dataset of labeled messages (spam/ham).

Data Cleaning – Removes special characters, punctuation, and stopwords.

Text Transformation – Converts text into numerical vectors using TF-IDF or CountVectorizer.

Model Training – Trains a classifier such as Naive Bayes or Logistic Regression.

Prediction – The model predicts whether new input text is spam or not.

Deployment – Hosted on Streamlit or Heroku for easy access.

📂 Project Structure
email-spam-classifier/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── EDA_and_Model.ipynb
│
├── app.py                # Streamlit app
├── model.pkl             # Saved trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── main.py               # Training and preprocessing script

🧪 Installation & Usage
1. Clone the repository
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

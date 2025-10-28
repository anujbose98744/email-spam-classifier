import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email/SMS Spam Classifier")
st.write("Check whether a message is spam or not. Try typing your own message or select a sample below:")

# Sample messages
sample_messages = [
    "Congratulations! You've won a $1000 gift card.",
    "Hey, are we still meeting tomorrow?",
    "Urgent! Your account will be locked if you don't respond.",
    "Lunch at 1 PM?",
    "Claim your prize now! Click the link."
]

st.subheader("Try a sample message:")
cols = st.columns(len(sample_messages))
for i, msg in enumerate(sample_messages):
    if cols[i].button(msg[:20] + "..."):  # show first 20 chars
        input_sms = msg
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")

# Text area for custom input
st.subheader("Or enter your own message:")
input_sms = st.text_area("Type your message here:")

if st.button("Predict"):
    if input_sms.strip() != "":
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a message first!")


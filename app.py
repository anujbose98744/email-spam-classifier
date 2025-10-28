import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# --- Page Configuration ---
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

# --- Custom Dark Theme CSS ---
st.markdown("""
    <style>
        /* Global background gradient */
        body {
            background: linear-gradient(135deg, #0f0f0f, #1c1c1c);
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Centered main container */
        .stApp {
            background: rgba(30, 30, 30, 0.85);
            color: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0px 4px 25px rgba(0, 0, 0, 0.5);
            max-width: 800px;
            margin: auto;
        }

        /* Headings */
        h1, h2, h3, h4 {
            text-align: center;
            color: #00d4ff;
        }

        /* Buttons (general style) */
        div.stButton > button {
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
        }

        /* Sample message buttons (black text) */
        .sample-btn > button {
            background-color: #f0f0f0 !important;
            color: #000000 !important;
            border: 2px solid #00d4ff !important;
        }

        .sample-btn > button:hover {
            background-color: #00d4ff !important;
            color: #000 !important;
            transform: scale(1.05);
        }

        /* Predict button (teal & bright) */
        .predict-btn > button {
            background-color: #00d4ff !important;
            color: #000 !important;
            border-radius: 10px;
            border: none;
            padding: 10px 25px;
            font-weight: bold;
            transition: 0.3s;
        }

        .predict-btn > button:hover {
            background-color: #1affd5 !important;
            transform: scale(1.05);
        }

        /* Text area */
        textarea {
            border-radius: 10px !important;
            border: 2px solid #00d4ff !important;
            background-color: #111 !important;
            color: white !important;
        }

        /* Streamlit info boxes */
        .stSuccess {
            background-color: rgba(0, 255, 128, 0.1);
        }
        .stError {
            background-color: rgba(255, 0, 0, 0.1);
        }
        .stWarning {
            background-color: rgba(255, 165, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Function to preprocess text ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# --- Load Model and Vectorizer ---
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- UI Layout ---
st.title("📧 Email/SMS Spam Classifier")
st.write("Check whether a message is spam or not — with a clean dark theme and bright buttons!")

# --- Sample messages ---
sample_messages = [
    "Congratulations! You've won a $1000 gift card.",
    "Hey, are we still meeting tomorrow?",
    "Urgent! Your account will be locked if you don't respond.",
    "Lunch at 1 PM?",
    "Claim your prize now! Click the link."
]

st.subheader("💬 Try a sample message:")
cols = st.columns(len(sample_messages))
for i, msg in enumerate(sample_messages):
    with cols[i]:
        if st.button(msg[:20] + "...", key=f"sample_{i}", help=msg, use_container_width=True):
            st.session_state["input_sms"] = msg

# --- Text area for custom input ---
st.subheader("✍️ Or enter your own message:")
input_sms = st.text_area("Type your message here:", key="input_sms")

# --- Predict Button ---
predict_col = st.container()
with predict_col:
    predict_clicked = st.button("🔮 Predict", key="predict", use_container_width=True, type="primary")

if predict_clicked:
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

import streamlit as st
import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

nltk.download('punkt_tab')

# Initialize stemmer
ps = PorterStemmer()


# ---------- Text preprocessing ----------
def transform_text(text: str) -> str:
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(words)

# ---------- Load trained model ----------
tfidf = pickle.load(open(r"models/vectorizer.pkl", "rb"))
model = pickle.load(open(r"models/model.pkl", "rb"))

# ---------- Sample messages ----------
sample_messages = {
    "Free ringtone": "Congratulations! You've won a free ringtone. Reply WIN to claim now!",
    "Lottery scam": "You have won $1,000,000! Claim your prize by sending your bank details.",
    "Promo offer": "Get 50% OFF on your next purchase. Use code SAVE50 at checkout.",
    "Normal chat": "Hey, are we still meeting for lunch tomorrow?",
    "Reminder": "Don't forget to bring your notebook to class.",
}

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["💬 Single Message", "📊 Analytics & Reports"])

# -------------------- Tab 1: Single Message --------------------
with tab1:
    st.title("📧 Email / SMS Spam Classifier")
    st.markdown("Predict if a single message is **Spam** or **Not Spam**.")
    
    choice = st.selectbox("Select a sample message (optional)", ["None"] + list(sample_messages.keys()))
    if choice != "None":
        input_sms = sample_messages[choice]
    else:
        input_sms = st.text_area("Or type your own message here:")

    if st.button("🔍 Predict", key="single_predict"):
        if not input_sms.strip():
            st.warning("⚠️ Please enter a message or select a sample.")
        else:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            pred = model.predict(vector_input)[0]
            confidence = model.predict_proba(vector_input).max()
            
            if pred == 1:
                st.error(f"🚨 This message is Spam (Confidence: {confidence:.2f})")
            else:
                st.success(f"✅ This message is Not Spam (Confidence: {confidence:.2f})")
            
            # Highlight spammy keywords
            spam_keywords = ['free','winner','won','prize','urgent','offer','click','buy','cheap','credit','loan','congrat','limited','risk-free','earn','money','unsubscribe','claim','guarantee']
            found_keywords = [w for w in spam_keywords if w in transformed_sms]
            if found_keywords:
                st.subheader("⚠️ Spammy keywords detected:")
                st.write(", ".join(found_keywords))
            else:
                st.subheader("No obvious spammy keywords detected.")

# -------------------- Tab 2: Analytics & Reports --------------------
with tab2:
    st.title("📊 Batch Analysis & Visualization")
    st.markdown("Upload a CSV file with a column named `text` containing messages to classify.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file,encoding='latin1')
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            # Transform and predict
            df['transformed_text'] = df['text'].apply(transform_text)
            X = tfidf.transform(df['transformed_text'])
            df['prediction'] = model.predict(X)
            df['confidence'] = model.predict_proba(X).max(axis=1)
            
            # Show statistics
            total = len(df)
            spam_count = df['prediction'].sum()
            ham_count = total - spam_count
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total messages", total)
            col2.metric("Spam messages", spam_count)
            col3.metric("Ham messages", ham_count)
            
            # --------- Most common spam keywords ---------
            st.subheader("Most common spam keywords in dataset")
            spam_keywords = ['free','winner','won','prize','urgent','offer','click','buy','cheap','credit','loan','congrat','limited','risk-free','earn','money','unsubscribe','claim','guarantee']
            all_text = " ".join(df.loc[df['prediction']==1, 'transformed_text'])
            found_keywords = [w for w in spam_keywords if w in all_text]

            if found_keywords:
                # Display in multiple columns if many keywords
                n_cols = 3
                cols = st.columns(n_cols)
                for i, keyword in enumerate(found_keywords):
                    cols[i % n_cols].write(f"- {keyword}")
            else:
                st.write("No common spam keywords found.")
            
           # --------- Visualizations in separate subtabs ---------
            vis_tab1, vis_tab2 = st.tabs(["📊 Pie Chart", "☁️ Word Cloud"])

            with vis_tab1:
                st.subheader("Spam vs Ham Pie Chart")
                fig, ax = plt.subplots()
                ax.pie([spam_count, ham_count], labels=["Spam","Ham"], autopct="%1.1f%%", colors=["red","green"])
                st.pyplot(fig)

            with vis_tab2:
                st.subheader("Word Cloud of Spam Messages")
                if all_text:
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
                    fig_wc, ax_wc = plt.subplots(figsize=(10,5))
                    ax_wc.imshow(wordcloud, interpolation="bilinear")
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                else:
                    st.write("No spam messages to generate word cloud.")
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv")

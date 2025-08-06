import streamlit as st
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

nlp = load_model()

# Streamlit app
st.title("💊 Drug Review Sentiment Analysis")
st.markdown("Analyze the sentiment of patient reviews for drugs using **DistilBERT**.")

# Single review input
st.subheader("📝 Analyze a Medicine Review")
user_input = st.text_area("Enter a drug review", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        result = nlp(user_input)[0]
        label = result['label']
        score = result['score'] * 100

        if label == 'POSITIVE':
            st.success(f"Sentiment: **{label}** \n\nConfidence: `{score:.2f}%`")
        else:
            st.error(f"Sentiment: **{label}** \n\nConfidence: `{score:.2f}%`")
    else:
        st.warning("Please enter a review to analyze.")

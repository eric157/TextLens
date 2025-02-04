import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import emoji
import spacy
from spacy import displacy

# --- Page Settings ---
st.set_page_config(
    page_title="TextLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lightweight Models Loading ---
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_ner_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- Core Analysis Functions ---
def analyze_sentiment(text):
    sentiment_result = load_sentiment_pipeline()(text[:512])
    return sentiment_result[0]

def extract_entities(text):
    nlp = load_ner_model()
    doc = nlp(text[:512])
    return [(ent.text, ent.label_) for ent in doc.ents]

def summarize_text(text):
    summary_result = load_summarization_pipeline()(text[:512], max_length=150, min_length=30)
    return summary_result[0]['summary_text']

def analyze_hashtags(text):
    return re.findall(r"#(\w+)", text[:512])

def analyze_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

def analyze_textblob_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# --- UI Components ---
def main():
    st.title("📊 TextLens - Lightweight Text Analysis")
    
    # Main text input
    text_input = st.text_area("Enter text for analysis:", height=150, 
                            help="Maximum 500 characters recommended")
    
    if text_input:
        with st.spinner('Analyzing text...'):
            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(text_input))
            with col2:
                st.metric("Words", len(text_input.split()))
            with col3:
                st.metric("Sentences", len(re.split(r'[.!?]+', text_input)))
            
            # Sentiment Analysis
            st.subheader("Sentiment Analysis")
            sentiment_result = analyze_sentiment(text_input)
            polarity, subjectivity = analyze_textblob_sentiment(text_input)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Main Sentiment", 
                        f"{sentiment_result['label']} ({sentiment_result['score']:.2f})")
            with col2:
                st.metric("TextBlob Metrics", 
                        f"Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")
            
            # Entity Recognition
            st.subheader("Named Entities")
            entities = extract_entities(text_input)
            if entities:
                st.json({label: [ent for ent, l in entities if l == label] 
                        for ent, label in set(entities)})
            else:
                st.write("No entities found")
            
            # Additional Features
            st.subheader("Text Insights")
            with st.expander("Summary"):
                st.write(summarize_text(text_input))
            
            with st.expander("Hashtags"):
                if hashtags := analyze_hashtags(text_input):
                    st.write(", ".join(f"#{tag}" for tag in hashtags))
                else:
                    st.write("No hashtags found")
            
            with st.expander("Emojis"):
                if emojis := analyze_emojis(text_input):
                    st.write(" ".join(emojis))
                else:
                    st.write("No emojis found")

if __name__ == "__main__":
    main()
import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import re
import emoji
from textblob import TextBlob
from streamlit_lottie import st_lottie
import json
import requests
from PIL import Image
import numpy as np
from bertopic import BERTopic
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from spacy import displacy
import plotly.express as px
from streamlit_extras.app_logo import add_logo
import time
from streamlit_option_menu import option_menu
import os

# --- Page Settings ---
st.set_page_config(
    page_title="TextLens",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Define Theme colors

LIGHT_MODE = {
 "primary_color":"#1E90FF",
  "background_color":"#ffffff",
  "text_color":"#000000",
  "secondary_background":"#f5f5f5",
   "grey_text": "#454545"
}
DARK_MODE = {
   "primary_color":"#1E90FF",
  "background_color":"#0E1117",
    "text_color":"#ffffff",
   "secondary_background":"#181818",
   "grey_text": "#919191"
}
DEFAULT_THEME = "dark"
#--- Functions
@st.cache_resource
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_emotion_pipeline():
     return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

@st.cache_resource
def load_keyword_pipeline():
     return pipeline("text2text-generation", model="google/flan-t5-base")


@st.cache_resource
def load_sarcasm_pipeline():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def load_topic_model():
    return BERTopic(embedding_model="all-MiniLM-L6-v2")

@st.cache_resource
def load_ner_pipeline():
     return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

@st.cache_resource
def load_absa_pipeline():
    return pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_resource
def load_toxicity_pipeline():
     return pipeline("text-classification", model="unitary/toxic-bert")

@st.cache_resource
def load_summarization_pipeline():
     return pipeline("summarization", model="facebook/bart-large-cnn")

def analyze_sentiment(text):
    sentiment_result = load_sentiment_pipeline()(text[:512])# Truncate the input here to max 512 tokens
    return sentiment_result[0]

def analyze_emotions(text):
    emotions_result = load_emotion_pipeline()(text[:512]) # Truncate the input here to max 512 tokens
    return emotions_result

def extract_keywords(text):
     keyword_pipeline = load_keyword_pipeline()
     prompt = f"Extract keywords: {text}"
     keywords_result = keyword_pipeline(prompt[:512],  max_length=50, num_return_sequences=1)
     return [res['generated_text'] for res in keywords_result]


def detect_sarcasm(text):
    sarcasm_result = load_sarcasm_pipeline()(text[:512]) # Truncate the input here to max 512 tokens
    # Modified code, return sarcasm if negative, otherwise its no sarcasm (labels available)
    if sarcasm_result[0]['label'] == 'Negative' and sarcasm_result[0]['score'] >0.5:
        return {'label': 'Sarcasm Detected', 'score': sarcasm_result[0]['score'] }
    return  {'label': 'No Sarcasm Detected', 'score': 1 - sarcasm_result[0]['score'] }

def generate_wordcloud(text, theme="light"):
    color =  "white" if theme == "light" else "#181818"
    wordcloud = WordCloud(width=800, height=400, background_color=color, colormap='Blues').generate(text[:512])# Truncate the input here to max 512 tokens
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt

def analyze_hashtags(text):
    hashtags = re.findall(r"#(\w+)", text[:512]) # Truncate the input here to max 512 tokens
    return hashtags

def analyze_emojis(text):
    emojis = [char for char in text if char in emoji.EMOJI_DATA]
    return emojis

def analyze_textblob_sentiment(text):
     analysis = TextBlob(text)
     return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def extract_topics(texts):
    topic_model = load_topic_model()
    if len(texts) < 2:
       try:
            topic_model.fit(texts)
            topics =  topic_model.get_document_info(texts)
       except:
            return pd.DataFrame()
    else:
        topics = topic_model.fit_transform(texts)[0]
        topics =  topic_model.get_document_info(texts)
    return topics

def display_download_button(df, file_format, filename):
    if not df.empty:
        if file_format == "csv":
             csv_buffer = io.StringIO()
             df.to_csv(csv_buffer, index=False)
             csv_content = csv_buffer.getvalue().encode('utf-8')
             st.download_button(label=f"Download {filename}.csv", data=csv_content, file_name=f"{filename}.csv", mime="text/csv")
        elif file_format == "json":
            json_content = df.to_json(orient="records").encode('utf-8')
            st.download_button(label=f"Download {filename}.json", data=json_content, file_name=f"{filename}.json", mime="application/json")

def extract_entities(text):
    ner_results = load_ner_pipeline()(text[:512])# Truncate the input here to max 512 tokens
    unique_entities = []
    seen = set()

    for entity in ner_results:
        entity_tuple = (entity['entity_group'], entity['start'], entity['end'], entity['word'])
        if entity_tuple not in seen:
            unique_entities.append(entity)
            seen.add(entity_tuple)
    return unique_entities

def highlight_entities(text, entities, theme="light"):
     highlighted_text = ""
     offset = 0
     if theme == 'light':
       entity_colors = {"PER": "#aaf", "ORG":"#afa", "LOC":"#faa", "MISC":"#afa", "GPE": "#ffa" }
     elif theme == 'dark':
          entity_colors = {"PER": "#3a3372", "ORG":"#3b4649", "LOC":"#851e0c", "MISC":"#334f5a", "GPE":"#967200" }
     else:
        entity_colors = {"PER": "#aaf", "ORG":"#afa", "LOC":"#faa", "MISC":"#afa", "GPE": "#ffa" }


     for entity in entities:
         start, end = entity['start'], entity['end']
         highlighted_text += text[offset:start]
         entity_type = entity['entity_group']
         if entity_type in ["PER", "ORG", "LOC", "MISC", "GPE"]:
             highlighted_text += f"<span class='entity-highlight' style='background-color: {entity_colors.get(entity_type)};'>{text[start:end]}</span>"
         else:
             highlighted_text += text[offset:]
         offset = end
     highlighted_text += text[offset:]
     return highlighted_text

def analyze_aspect_sentiment(text, aspect):
    result = load_absa_pipeline()(f"{aspect} {text[:512]}")# Truncate the input here to max 512 tokens
    return result[0]

def analyze_sentence_sentiment(text):
     sentences = sent_tokenize(text)
     results = []
     for sentence in sentences:
         sentiment = analyze_sentiment(sentence[:512])# Truncate the input here to max 512 tokens
         results.append({"sentence": sentence, "sentiment": sentiment})
     return results

def detect_toxicity(text):
        toxicity_results = load_toxicity_pipeline()(text[:512])# Truncate the input here to max 512 tokens
        return toxicity_results[0]

def summarize_text(text):
    summary_result = load_summarization_pipeline()(text[:512])# Truncate the input here to max 512 tokens
    return summary_result[0]['summary_text']

def use_app_theme(theme):
    colors = DARK_MODE if theme =="dark" else LIGHT_MODE
    st.markdown(
    f"""
      <style>
          .reportview-container .main .block-container{{
            background-color: {colors.get('background_color')};
           color: {colors.get('text_color')};
        }}

        .st-emotion-cache-10tr677{{
          color:  {colors.get('text_color')};
         }}
        .st-emotion-cache-1x63k98{{
            color: {colors.get('grey_text')};
        }}
        .st-emotion-cache-1v0mbdr{{
            color:  {colors.get('text_color')};
         }}
         .entity-highlight {{
           color:  {colors.get('text_color')};
          }}
        .st-emotion-cache-r421ms {{
              background-color: {colors.get('secondary_background')};
         }}
       .st-emotion-cache-1dm1mcz{{
             background-color:{colors.get('secondary_background')};
         }}
       .st-emotion-cache-1b2590p{{
             background-color: {colors.get('secondary_background')};
        }}
       [data-baseweb="base-input"]{{
           background-color:{colors.get('secondary_background')};
         color:{colors.get('text_color')};
       }}
      textarea[data-streamlit="text-area"]{{
          color:  #FFFFFF !important;
        }}
      textarea[data-streamlit="text-area"]::placeholder{{
        color: #FFFFFF !important;
            opacity: 0.8;
        }}

     [data-testid='stImage'] > div > img {{
          border-radius: 10px;
         max-width: 200px;
     }}
     #word-count {{
     font-size: 0.8rem;
     color: #999;
     }}


      </style>
      """, unsafe_allow_html=True,
    )

# Define the relative path for the logo
logo_path = "logo.png"


# Verify logo path before applying
try:
    with Image.open(logo_path) as logo:
        st.sidebar.image(logo, width=150, use_container_width=True)
        st.markdown(""" <style> [data-testid='stImage'] > div > img {  border-radius: 10px}</style>""", unsafe_allow_html=True)
except Exception as e:
       st.sidebar.error(f"Error Loading Logo: {e}", icon="🚨")

# ---- Sidebar Content ---
with st.sidebar:
        st.title("⚙️ Settings & Info")
        st.markdown("---")
        st.subheader("📌 About the App")
        st.write("Perform advanced text analysis including sentiment, emotion, topic modeling, and more.")
        st.markdown("---")
        with st.expander("💡 Model Details"):
              st.write("This app leverages multiple pre-trained transformer models from the Hugging Face Transformers library. These models are continuously being improved and updated. Model selection may have impact on both accuracy and performance, and can be changed in the code.")
        st.markdown("---")
        with st.expander("📖 Usage Guide"):
              st.markdown("""
            **Text Analysis:**
            - Enter text in the text box for real-time analysis.
            **File Upload & Batch Processing:**
               - Upload .csv or .txt files for batch processing. Ensure the csv file has a text column.
             - Results can be downloaded.
           **Advanced Analysis:**
             - Additional analysis such as NER, ABSA, Sentence Analysis, Summarization are also available.
           - The Text limit is 300 words
          """)
        st.markdown("---")

use_app_theme(DEFAULT_THEME)
#add_logo("logo.png", height=50) - now this image is set correctly based on other configuration from above for both side and webpage favicons in browser UI. This ensures now that everything gets rendered

# ---- Main App Content ----
lottie_url = 'https://lottie.host/8ef588a6-1e2f-4797-9c06-1655b9253efb/zFj7X4kX6J.json'
lottie_json = load_lottieurl(lottie_url)
if lottie_json:
    st_lottie(lottie_json, speed=1, height=180, quality='high')

# Title styling
st.markdown(f"""
    <h1 style='text-align: center; color: {DARK_MODE['primary_color']}; font-size: 2.5em; margin-bottom: 0.5em; font-weight: 700;'>
        TextLens – Analyze text with clarity
    </h1>
""", unsafe_allow_html=True)

MAX_WORDS = 300 # setting max value of inputs

tab1, tab2, tab3 = st.tabs(["Text Analysis", "File Upload", "Visualization & Reports"])
theme = DEFAULT_THEME

with tab1: # Text Analysis
      st.header("Text Analysis")
    # add UI so users better understand usage with explicit mention
      st.markdown(f"Enter text for analysis (Maximum: {MAX_WORDS} words):", unsafe_allow_html=True)
      text_input = st.text_area("Enter text for analysis:", height=150,  key="text_area",  max_chars= MAX_WORDS * 7) # this assumes that every text is around has 7 character including space
      word_count = len(text_input.split())
      st.markdown(f'<div id="word-count">Word count: {word_count}</div>', unsafe_allow_html=True)  # show words limit


      enable_ner = st.checkbox("Enable Named Entity Recognition")
      enable_absa = st.checkbox("Enable Aspect-Based Sentiment Analysis")
      absa_aspect = st.text_input("Enter aspect for ABSA (e.g., battery life):", disabled=not enable_absa)
      enable_sentence_analysis = st.checkbox("Enable Sentence-Level Sentiment Analysis")
      enable_toxicity_detection = st.checkbox("Enable Toxicity Detection")
      enable_summarization = st.checkbox("Enable Text Summarization")

      if text_input:
        with st.spinner('Processing the text...'):
           sentiment_result = analyze_sentiment(text_input)
           emotion_result = analyze_emotions(text_input)
           keywords = extract_keywords(text_input)
           sarcasm_result = detect_sarcasm(text_input)
           hashtags = analyze_hashtags(text_input)
           emojis = analyze_emojis(text_input)
           textblob_sentiment = analyze_textblob_sentiment(text_input)

            # --- Display Results ---
           col1, col2 = st.columns(2)
           with col1:
               st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>📊 Sentiment Analysis</h3>", unsafe_allow_html=True)
               st.metric("Sentiment", value=sentiment_result['label'], delta=sentiment_result['score'])

               with st.expander("📌 TextBlob Sentiment Analysis"):
                  st.metric("Polarity", value=round(textblob_sentiment[0],2))
                  st.metric("Subjectivity", value=round(textblob_sentiment[1],2))

           with col2:
              st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>💖 Emotion Classification</h3>", unsafe_allow_html=True)
              for emotion in emotion_result:
                 st.metric(emotion['label'], value=round(emotion['score'],2))

           st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>🤔 Sarcasm Detection</h3>", unsafe_allow_html=True)
           st.metric("Sarcasm", value=sarcasm_result['label'], delta=round(sarcasm_result['score'],2))
           st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>🔑 Keyword Extraction</h3>", unsafe_allow_html=True)
           st.write(", ".join(keywords))

           if hashtags:
              st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>#️⃣ Hashtags</h3>", unsafe_allow_html=True)
              st.write(", ".join(hashtags))

           if emojis:
               st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>😀 Emojis</h3>", unsafe_allow_html=True)
               st.write(" ".join(emojis))

           st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'> ☁️ Word Cloud Visualization</h3>", unsafe_allow_html=True)
           wordcloud_fig = generate_wordcloud(text_input, theme=theme)
           st.pyplot(wordcloud_fig)

           if enable_ner:
                st.subheader("Named Entity Recognition (NER)")
                entities = extract_entities(text_input)
                highlighted_text = highlight_entities(text_input, entities, theme=theme)
                st.markdown(highlighted_text, unsafe_allow_html=True)

           if enable_absa and absa_aspect:
               st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>❤️‍🔥 Aspect-Based Sentiment Analysis (ABSA)</h3>", unsafe_allow_html=True)
               absa_result = analyze_aspect_sentiment(text_input, absa_aspect)
               st.metric(f"Sentiment towards '{absa_aspect}'", value=absa_result['label'], delta=absa_result['score'])

           if enable_sentence_analysis:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'> 🔍 Sentence-Level Sentiment Analysis</h3>", unsafe_allow_html=True)
                sentence_sentiments = analyze_sentence_sentiment(text_input)
                for item in sentence_sentiments:
                   st.markdown(f"- **Sentence:** {item['sentence']}")
                   st.markdown(f"  **Sentiment:** {item['sentiment']['label']} (Score: {item['sentiment']['score']:.2f})")

           if enable_toxicity_detection:
                toxicity_result = detect_toxicity(text_input)
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'> 🚨 Toxicity & Hate Speech Detection</h3>", unsafe_allow_html=True)
                st.markdown(f"**Toxicity Level:** {toxicity_result['label']} (Score: {toxicity_result['score']:.2f})")
                if toxicity_result['label'] == "toxic":
                      st.warning("Warning: The text contains potentially toxic content.", icon="🚨")

           if enable_summarization:
                 st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'> 📄 AI-Powered Text Summarization</h3>", unsafe_allow_html=True)
                 summary = summarize_text(text_input)
                 st.write(summary)

with tab2:  # File Upload
    st.header("File Upload & Batch Processing")
    uploaded_file = st.file_uploader("Drag & Drop CSV/TXT file here", type=["csv", "txt"], accept_multiple_files=False)

    if uploaded_file:
        with st.spinner('Processing the file...'):
            if uploaded_file.name.endswith(".csv"):
                try:
                     df = pd.read_csv(uploaded_file)
                     if 'text' not in df.columns:
                        st.error("CSV file must have a 'text' column.", icon="🚨")
                        st.stop()
                     texts = df['text'].tolist()
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}", icon="🚨")
                    st.stop()

            elif uploaded_file.name.endswith(".txt"):
                try:
                    text_file = uploaded_file.read().decode('utf-8')
                    texts = [text_file]
                except Exception as e:
                    st.error(f"Error reading TXT file: {e}", icon="🚨")
                    st.stop()

            sentiments = [analyze_sentiment(text) for text in texts]
            emotions = [analyze_emotions(text) for text in texts]
            keywords_list = [extract_keywords(text) for text in texts]
            sarcasm_results = [detect_sarcasm(text) for text in translated_texts]
            textblob_sentiments = [analyze_textblob_sentiment(text) for text in texts]

            # --- Process TextBlob Analysis ---
            textblob_polarity = [sentiment[0] for sentiment in textblob_sentiments]
            textblob_subjectivity = [sentiment[1] for sentiment in textblob_sentiments]

            # --- Process Sentiment Analysis ---
            sentiment_labels = [sentiment['label'] for sentiment in sentiments]
            sentiment_scores = [sentiment['score'] for sentiment in sentiments]

            # --- Process Emotion Analysis ---
            emotion_labels = []
            for emotion_list in emotions:
                emotion_labels.append(emotion_list[0]['label'] if emotion_list else 'No Emotion')
            emotion_scores = []
            for emotion_list in emotions:
                emotion_scores.append(emotion_list[0]['score'] if emotion_list else 0)

            # --- Process Sarcasm Detection ---
            sarcasm_labels = [sarcasm['label'] for sarcasm in sarcasm_results]
            sarcasm_scores = [sarcasm['score'] for sarcasm in sarcasm_results]


            if uploaded_file.name.endswith(".csv"):
                 df['sentiment'] = sentiment_labels
                 df['sentiment_score'] = sentiment_scores
                 df['emotion'] = emotion_labels
                 df['emotion_score'] = emotion_scores
                 df['keywords'] = [", ".join(keywords) for keywords in keywords_list]
                 df['sarcasm'] = sarcasm_labels
                 df['sarcasm_score'] = sarcasm_scores
                 df['textblob_polarity'] = textblob_polarity
                 df['textblob_subjectivity'] = textblob_subjectivity
            elif uploaded_file.name.endswith(".txt"):
                df = pd.DataFrame({
                 'text': texts,
                 'sentiment': sentiment_labels,
                 'sentiment_score': sentiment_scores,
                 'emotion': emotion_labels,
                 'emotion_score': emotion_scores,
                 'keywords': [", ".join(keywords) for keywords in keywords_list],
                 'sarcasm': sarcasm_labels,
                 'sarcasm_score': sarcasm_scores,
                 'textblob_polarity': textblob_polarity,
                 'textblob_subjectivity': textblob_subjectivity
                })

            st.subheader("Analysis Results")
            st.dataframe(df, height=300)

            st.subheader("Topics Extraction")
            topics_df = extract_topics(translated_texts)
            if not topics_df.empty:
                st.dataframe(topics_df[['Document', 'Topic', 'Name']], height=300)
                with st.expander('Detailed Topics'):
                    st.dataframe(topics_df.drop('Representative_Docs', axis=1), height=300)

with tab3: # Visualization & Reports
    if uploaded_file:
        st.header("Visualization & Reports")
        if not df.empty:
                col1, col2 = st.columns(2)
                with col1:
                   st.subheader("Download Results as CSV")
                   display_download_button(df, "csv", "analysis_results")
                with col2:
                    st.subheader("Download Results as JSON")
                    display_download_button(df, "json", "analysis_results")
                if not df.empty:
                    sentiment_counts = df['sentiment'].value_counts()
                    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                                title="Sentiment Distribution",
                                color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig)
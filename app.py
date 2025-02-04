import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import re
import emoji
from textblob import TextBlob  # Simplified Sentiment
from streamlit_lottie import st_lottie
import json
import requests
from PIL import Image
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
    "primary_color": "#1E90FF",
    "background_color": "#ffffff",
    "text_color": "#000000",
    "secondary_background": "#f5f5f5",
    "grey_text": "#454545"
}
DARK_MODE = {
    "primary_color": "#1E90FF",
    "background_color": "#0E1117",
    "text_color": "#ffffff",
    "secondary_background": "#181818",
    "grey_text": "#919191"
}
DEFAULT_THEME = "dark"

# --- Functions ---
@st.cache_resource
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def analyze_textblob_sentiment(text):  # Simple Sentiment Analysis
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def generate_wordcloud(text, theme="light"):
    color = "white" if theme == "light" else "#181818"
    wordcloud = WordCloud(width=800, height=400, background_color=color, colormap='Blues').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt

def analyze_hashtags(text):
    hashtags = re.findall(r"#(\w+)", text)
    return hashtags

def analyze_emojis(text):
    emojis = [char for char in text if char in emoji.EMOJI_DATA]
    return emojis

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

def use_app_theme(theme):
    colors = DARK_MODE if theme == "dark" else LIGHT_MODE
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
    st.write("Perform text analysis including sentiment, and more.")
    st.markdown("---")
    st.markdown("---")
    with st.expander("📖 Usage Guide"):
        st.markdown("""
            **Text Analysis:**
            - Enter text in the text box for real-time analysis.
            **File Upload & Batch Processing:**
               - Upload .csv or .txt files for batch processing. Ensure the text column.
             - Results can be downloaded.
           **Advanced Analysis:**
             - Additional analysis such as NER, ABSA, Summarization are also available.
           - The Text limit is 300 words
          """)
    st.markdown("---")

use_app_theme(DEFAULT_THEME)
#add_logo("logo.png", height=50)

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

MAX_WORDS = 300

tab1, tab2, tab3 = st.tabs(["Text Analysis", "File Upload", "Visualization & Reports"])
theme = DEFAULT_THEME

with tab1:  # Text Analysis
    st.header("Text Analysis")
    st.markdown(f"Enter text for analysis (Maximum: {MAX_WORDS} words):", unsafe_allow_html=True)
    text_input = st.text_area("Enter text for analysis:", height=150, key="text_area", max_chars=MAX_WORDS * 7)
    word_count = len(text_input.split())
    st.markdown(f'<div id="word-count">Word count: {word_count}</div>', unsafe_allow_html=True)  # show words limit

    if text_input:
        with st.spinner('Processing the text...'):
            textblob_sentiment = analyze_textblob_sentiment(text_input)
            hashtags = analyze_hashtags(text_input)
            emojis = analyze_emojis(text_input)

            # --- Display Results ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>📊 Sentiment Analysis</h3>", unsafe_allow_html=True)
                st.metric("Polarity", value=round(textblob_sentiment[0], 2))
                st.metric("Subjectivity", value=round(textblob_sentiment[1], 2))

            with col2:
                if hashtags:
                    st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>#️⃣ Hashtags</h3>", unsafe_allow_html=True)
                    st.write(", ".join(hashtags))

                if emojis:
                    st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>😀 Emojis</h3>", unsafe_allow_html=True)
                    st.write(" ".join(emojis))

            st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'> ☁️ Word Cloud Visualization</h3>", unsafe_allow_html=True)
            wordcloud_fig = generate_wordcloud(text_input, theme=theme)
            st.pyplot(wordcloud_fig)

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

            textblob_sentiments = [analyze_textblob_sentiment(text) for text in texts]

            # --- Process TextBlob Analysis ---
            textblob_polarity = [sentiment[0] for sentiment in textblob_sentiments]
            textblob_subjectivity = [sentiment[1] for sentiment in textblob_sentiments]

            if uploaded_file.name.endswith(".csv"):
                df['textblob_polarity'] = textblob_polarity
                df['textblob_subjectivity'] = textblob_subjectivity
            elif uploaded_file.name.endswith(".txt"):
                df = pd.DataFrame({
                    'text': texts,
                    'textblob_polarity': textblob_polarity,
                    'textblob_subjectivity': textblob_subjectivity
                })

            st.subheader("Analysis Results")
            st.dataframe(df, height=300)

with tab3:  # Visualization & Reports
    if uploaded_file and not df.empty:
        st.header("Visualization & Reports")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Download Results as CSV")
            display_download_button(df, "csv", "analysis_results")
        with col2:
            st.subheader("Download Results as JSON")
            display_download_button(df, "json", "analysis_results")

        sentiment_counts = pd.cut(df['textblob_polarity'], bins=5, labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']).value_counts() #binning it to group sentiment to the scale of  'Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'
        fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig)
import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import re
import emoji
from textblob import TextBlob
import requests
from PIL import Image
import plotly.express as px
from streamlit_extras.app_logo import add_logo
from streamlit_option_menu import option_menu
import os

# --- Page Settings ---
st.set_page_config(
    page_title="TextLens",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Theme colors (simplified)
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
# Removed Lottie Loading Function

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_emotion_pipeline():
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

@st.cache_resource
def load_keyword_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")


def analyze_sentiment(text):
    try:
        sentiment_result = load_sentiment_pipeline()(text[:512])
        return sentiment_result[0]
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return {"label": "Error", "score": 0.0}


def analyze_emotions(text):
    try:
        emotions_result = load_emotion_pipeline()(text[:512])
        return emotions_result
    except Exception as e:
        st.error(f"Error during emotion analysis: {e}")
        return [{"label": "Error", "score": 0.0}]

def extract_keywords(text):
    try:
        keyword_pipeline = load_keyword_pipeline()
        prompt = f"Extract keywords: {text}"
        keywords_result = keyword_pipeline(prompt[:512],  max_length=50, num_return_sequences=1)
        return [res['generated_text'] for res in keywords_result]
    except Exception as e:
        st.error(f"Error during keyword extraction: {e}")
        return []


def generate_wordcloud(text, theme="light"):
    color = "white" if theme == "light" else "#181818"
    try:
        wordcloud = WordCloud(width=800, height=400, background_color=color, colormap='viridis').generate(text[:512])  # Use viridis colormap
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        return plt
    except Exception as e:
        st.error(f"Error during wordcloud generation: {e}")
        return None  # Or some other placeholder

def analyze_hashtags(text):
    try:
        hashtags = re.findall(r"#(\w+)", text[:512])
        return hashtags
    except Exception as e:
        st.error(f"Error during hashtag analysis: {e}")
        return []

def analyze_emojis(text):
    try:
        emojis = [char for char in text if char in emoji.EMOJI_DATA]
        return emojis
    except Exception as e:
        st.error(f"Error during emoji analysis: {e}")
        return []

def analyze_textblob_sentiment(text):
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    except Exception as e:
        st.error(f"Error during TextBlob sentiment analysis: {e}")
        return 0.0, 0.0

def estimate_reading_time(text):
    words = len(text.split())
    reading_speed_wpm = 200  # Average reading speed in words per minute
    reading_time = words / reading_speed_wpm
    return round(reading_time, 2)

def calculate_lexical_diversity(text):
    words = text.split()
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def count_sentences(text):
    sentences = re.split(r'[.!?]+', text)  # Splitting by common sentence delimiters
    return len(sentences) - 1 if sentences else 0  # Adjust count to exclude empty strings

def analyze_average_word_length(text):
    words = text.split()
    if not words:
        return 0.0
    total_length = sum(len(word) for word in words)
    return total_length / len(words)

def display_download_button(df, file_format, filename):
    if not df.empty:
        try:
            if file_format == "csv":
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue().encode('utf-8')
                st.download_button(label=f"Download {filename}.csv", data=csv_content, file_name=f"{filename}.csv",
                                    mime="text/csv")
            elif file_format == "json":
                json_content = df.to_json(orient="records").encode('utf-8')
                st.download_button(label=f"Download {filename}.json", data=json_content, file_name=f"{filename}.json",
                                    mime="application/json")
        except Exception as e:
            st.error(f"Error during download button creation: {e}")

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
        st.markdown(""" <style> [data-testid='stImage'] > div > img {  border-radius: 10px}</style>""",
                    unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error(f"Error Loading Logo: {e}", icon="🚨")

# ---- Sidebar Content ---
with st.sidebar:
    st.title("⚙️ Settings & Info")
    st.markdown("---")
    st.subheader("📌 About the App")
    st.write("Perform text analysis including sentiment, emotion, and keyword extraction.")
    st.markdown("---")
    with st.expander("💡 Model Details"):
        st.write("This app leverages pre-trained transformer models from the Hugging Face Transformers library.")
    st.markdown("---")
    with st.expander("📖 Usage Guide"):
        st.markdown("""
            **Text Analysis:**
            - Enter text in the text box for real-time analysis.
            **File Upload & Batch Processing:**
               - Upload .csv or .txt files for batch processing. Ensure the text column.
             - Results can be downloaded.
          """)
    st.markdown("---")

use_app_theme(DEFAULT_THEME)

# ---- Main App Content ----

# Title styling
st.markdown(f"""
    <h1 style='text-align: center; color: {DARK_MODE['primary_color']}; font-size: 2.5em; margin-bottom: 0.5em; font-weight: 700;'>
        TextLens – Analyze text with clarity
    </h1>
""", unsafe_allow_html=True)

MAX_WORDS = 300  # setting max value of inputs

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
            sentiment_result = analyze_sentiment(text_input)
            emotion_result = analyze_emotions(text_input)
            keywords = extract_keywords(text_input)
            hashtags = analyze_hashtags(text_input)
            emojis = analyze_emojis(text_input)
            textblob_sentiment = analyze_textblob_sentiment(text_input)

            # --- Additional Minimalist Features ---
            reading_time = estimate_reading_time(text_input)
            lexical_diversity = calculate_lexical_diversity(text_input)
            sentence_count = count_sentences(text_input)
            avg_word_length = analyze_average_word_length(text_input)

            # --- Display Results ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>📊 Sentiment Analysis</h3>",
                            unsafe_allow_html=True)
                if sentiment_result['label'] == "Error":
                    st.error("Sentiment Analysis Failed")
                else:
                    st.metric("Sentiment", value=sentiment_result['label'], delta=sentiment_result['score'])

                with st.expander("📌 TextBlob Sentiment Analysis"):
                    st.metric("Polarity", value=round(textblob_sentiment[0], 2))
                    st.metric("Subjectivity", value=round(textblob_sentiment[1], 2))

            with col2:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>💖 Emotion Classification</h3>",
                            unsafe_allow_html=True)
                for emotion in emotion_result:
                    st.metric(emotion['label'], value=round(emotion['score'], 2))

            st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>🔑 Keyword Extraction</h3>",
                        unsafe_allow_html=True)
            st.write(", ".join(keywords))

            if hashtags:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>#️⃣ Hashtags</h3>", unsafe_allow_html=True)
                st.write(", ".join(hashtags))

            if emojis:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>😀 Emojis</h3>", unsafe_allow_html=True)
                st.write(" ".join(emojis))

            # --- Display additional metrics ---
            col3, col4, col5, col6 = st.columns(4)
            with col3:
                st.metric("Reading Time (mins)", reading_time)
            with col4:
                st.metric("Lexical Diversity", round(lexical_diversity, 2))
            with col5:
                st.metric("Sentence Count", sentence_count)
            with col6:
                st.metric("Avg. Word Length", round(avg_word_length, 2))

            st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'> ☁️ Word Cloud Visualization</h3>",
                        unsafe_allow_html=True)
            wordcloud_fig = generate_wordcloud(text_input, theme=theme)
            if wordcloud_fig:  # Check if the figure was successfully created
                st.pyplot(wordcloud_fig)

with tab2:  # File Upload
    st.header("File Upload & Batch Processing")
    uploaded_file = st.file_uploader("Drag & Drop CSV/TXT file here", type=["csv", "txt"], accept_multiple_files=False)

    if uploaded_file:
        with st.spinner('Processing the file...'):
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                    if 'text' not in df.columns:
                        st.error("CSV file must have a 'text' column.", icon="🚨")
                        st.stop()
                    texts = df['text'].tolist()

                elif uploaded_file.name.endswith(".txt"):
                    text_file = uploaded_file.read().decode('utf-8')
                    texts = [text_file]
                else:
                    st.error("Unsupported file type. Please upload a CSV or TXT file.", icon="🚨")
                    st.stop()

                # --- Perform Analysis ---
                sentiments = []
                emotions = []
                keywords_list = []
                textblob_sentiments = []
                reading_times = []
                lexical_diversities = []
                sentence_counts = []
                avg_word_lengths = []

                for text in texts:
                    sentiments.append(analyze_sentiment(text))
                    emotions.append(analyze_emotions(text))
                    keywords_list.append(extract_keywords(text))
                    textblob_sentiments.append(analyze_textblob_sentiment(text))
                    reading_times.append(estimate_reading_time(text))
                    lexical_diversities.append(calculate_lexical_diversity(text))
                    sentence_counts.append(count_sentences(text))
                    avg_word_lengths.append(analyze_average_word_length(text))

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

                # --- Create DataFrame ---
                if uploaded_file.name.endswith(".csv"):
                    df['sentiment'] = sentiment_labels
                    df['sentiment_score'] = sentiment_scores
                    df['emotion'] = emotion_labels
                    df['emotion_score'] = emotion_scores
                    df['keywords'] = [", ".join(keywords) for keywords in keywords_list]
                    df['textblob_polarity'] = textblob_polarity
                    df['textblob_subjectivity'] = textblob_subjectivity
                    df['reading_time'] = reading_times
                    df['lexical_diversity'] = lexical_diversities
                    df['sentence_count'] = sentence_counts
                    df['avg_word_length'] = avg_word_lengths

                elif uploaded_file.name.endswith(".txt"):
                    df = pd.DataFrame({
                        'text': texts,
                        'sentiment': sentiment_labels,
                        'sentiment_score': sentiment_scores,
                        'emotion': emotion_labels,
                        'emotion_score': emotion_scores,
                        'keywords': [", ".join(keywords) for keywords in keywords_list],
                        'textblob_polarity': textblob_polarity,
                        'textblob_subjectivity': textblob_subjectivity,
                        'reading_time': reading_times,
                        'lexical_diversity': lexical_diversities,
                        'sentence_count': sentence_counts,
                        'avg_word_length': avg_word_lengths
                    })

                st.subheader("Analysis Results")
                st.dataframe(df, height=300)

            except Exception as e:
                st.error(f"An error occurred during file processing: {e}", icon="🚨")

with tab3:  # Visualization & Reports
    if uploaded_file and 'df' in locals() and not df.empty:
        st.header("Visualization & Reports")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Download Results as CSV")
            display_download_button(df, "csv", "analysis_results")
        with col2:
            st.subheader("Download Results as JSON")
            display_download_button(df, "json", "analysis_results")

        # Sentiment Distribution Pie Chart
        sentiment_counts = df['sentiment'].value_counts()
        fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                                title="Sentiment Distribution",
                                color_discrete_sequence=px.colors.sequential.Plasma)  # Cool color scheme
        st.plotly_chart(fig_sentiment)

        # Emotion Distribution Bar Chart
        emotion_counts = df['emotion'].value_counts()
        fig_emotion = px.bar(emotion_counts, x=emotion_counts.index, y=emotion_counts.values,
                             title="Emotion Distribution",
                             color_discrete_sequence=px.colors.sequential.Viridis)  # Another color scheme
        fig_emotion.update_layout(xaxis_title="Emotion", yaxis_title="Count")
        st.plotly_chart(fig_emotion)

        # Reading Time Distribution Histogram (Example)
        fig_reading_time = px.histogram(df, x="reading_time", title="Reading Time Distribution",
                                        color_discrete_sequence=px.colors.sequential.Cividis)  # Distribution Chart for Reading Time
        fig_reading_time.update_layout(xaxis_title="Reading Time (minutes)", yaxis_title="Frequency")
        st.plotly_chart(fig_reading_time)


    elif uploaded_file:
        st.warning("No data available to visualize. Ensure file processing was successful.")
    else:
        st.info("Upload a file to view visualizations.")
import streamlit as st
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
from collections import Counter
from textstat import flesch_reading_ease
from heapq import nlargest
import string

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

def analyze_sentiment(text):
   try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.1:
            return {"label": "Positive", "score": polarity}
        elif polarity < -0.1:
            return {"label": "Negative", "score": polarity}
        else:
            return {"label": "Neutral", "score": polarity}
   except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return {"label": "Error", "score": 0.0}

#No emotion analysis as it used transformer pipeline
def analyze_emotions(text):
    return {"label": "Not Available", "score": 0.0}

def extract_keywords(text):
    try:
        # Simple keyword extraction based on frequency
        text = text.lower()
        text = re.sub('[' + string.punctuation + ']', '', text)  # Remove punctuation
        words = text.split()
        stopwords = set([
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
            "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        ])
        filtered_words = [word for word in words if word not in stopwords]
        keyword_counts = Counter(filtered_words).most_common(10)
        keywords = [word for word, count in keyword_counts]
        return keywords
    except Exception as e:
        st.error(f"Error during keyword extraction: {e}")
        return []

def generate_wordcloud(text, theme="light"):
    color = "white" if theme == "light" else "#181818"
    try:
        wordcloud = WordCloud(width=800, height=400, background_color=color, colormap='viridis').generate(text[:512])
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        return plt
    except Exception as e:
        st.error(f"Error during wordcloud generation: {e}")
        return None

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
    reading_speed_wpm = 200
    reading_time = words / reading_speed_wpm
    return round(reading_time, 2)

def calculate_lexical_diversity(text):
    words = text.split()
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def count_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    return len(sentences) - 1 if sentences else 0

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
                st.download_button(label=f"Download {filename}.json", data=csv_content, file_name=f"{filename}.json",
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
         max_width: 200px;
     }}
      </style>
      """, unsafe_allow_html=True,
    )

# --- New Feature Functions ---

def extract_named_entities(text):
    return [] # Placeholder, as we're not using spaCy

def analyze_pos_tags(text):
   return {} # Placeholder, as we're not using spaCy

def calculate_stopword_density(text):
    text = text.lower()
    text = re.sub('[' + string.punctuation + ']', '', text)  # Remove punctuation
    words = text.split()
    stopwords = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])
    total_words = len(words)
    if total_words == 0:
        return 0.0
    stopword_count = len([w for w in words if w in stopwords])
    return stopword_count / total_words

def detect_passive_voice(text):
    return []  # Placeholder, requires more complex parsing

def count_pronouns(text):
    text = text.lower()
    words = text.split()
    pronoun_count = Counter(word for word in words if word in [
            "i", "me", "my", "mine", "we", "us", "our", "ours",
            "you", "your", "yours", "he", "him", "his", "she",
            "her", "hers", "it", "its", "they", "them", "their", "theirs"
        ])
    return pronoun_count

def count_first_third_person(text):
   text = text.lower()
   words = text.split()
   first_person_count = sum(1 for word in words if word in ["i", "me", "my", "mine", "we", "us", "our", "ours"])
   third_person_count = sum(1 for word in words if word in ["he", "him", "his", "she", "her", "hers", "it", "its"])
   return first_person_count, third_person_count

def correct_grammar(text):
     return str(TextBlob(text).correct())

def count_transition_words(text):
    text = text.lower()
    words = text.split()
    transition_words = ["however", "therefore", "in addition", "moreover", "consequently", "as a result", "for example"]
    count = sum(1 for word in words if word in transition_words)
    return count

# Replaced sumy summarization with a simple extractive summarization
def summarize_text(text, num_sentences=3):
   sentences = re.split(r'[.!?]+', text) # Split by common sentence delimiters
   sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings

   # Calculate sentence scores based on keyword frequency
   keywords = extract_keywords(text)
   sentence_scores = {}
   for i, sentence in enumerate(sentences):
        sentence_scores[i] = 0
        for word in keywords:
            if word in sentence.lower():
                sentence_scores[i] += 1

   # Select top N sentences with highest scores
   top_sentences_idx = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
   top_sentences = [sentences[i] for i in sorted(top_sentences_idx) if i < len(sentences)]  # Check for index validity!

   return " ".join(top_sentences)

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
    st.write("Perform text analysis including sentiment and keyword extraction.")
    st.markdown("---")
    with st.expander("💡 Model Details"):
        st.write("This app uses basic text processing techniques.")
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

MAX_WORDS = 300

tab1, tab2, tab3 = st.tabs(["Text Analysis", "File Upload", "Visualization & Reports"])
theme = DEFAULT_THEME

with tab1:  # Text Analysis
    st.header("Text Analysis")
    st.markdown(f"Enter text for analysis (Maximum: {MAX_WORDS} words):", unsafe_allow_html=True)
    text_input = st.text_area("Enter text for analysis:", height=150, key="text_area", max_chars=MAX_WORDS * 7)
    word_count = len(text_input.split())
    st.markdown(f'<div id="word-count">Word count: {word_count}</div>', unsafe_allow_html=True)

    if text_input:
        with st.spinner('Processing the text...'):
            # --- Core analyses ---
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

            # --- Newly Implemented Features ---
            named_entities = extract_named_entities(text_input)
            pos_counts = analyze_pos_tags(text_input)
            stopword_density = calculate_stopword_density(text_input)
            passive_sentences = detect_passive_voice(text_input)
            pronoun_counts = count_pronouns(text_input)
            first_person_count, third_person_count = count_first_third_person(text_input)
            #keyword_counts = count_keywords(text_input) #not in used and remove it
            corrected_text = correct_grammar(text_input)
            transition_word_count = count_transition_words(text_input)
            text_summary = summarize_text(text_input)

            # --- Display Results ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>📊 Sentiment Analysis</h3>",
                            unsafe_allow_html=True)
                if sentiment_result['label'] == "Error":
                    st.error("Sentiment Analysis Failed")
                else:
                    st.metric("Sentiment", value=sentiment_result['label'], delta=round(sentiment_result['score'], 2))

                with st.expander("📌 TextBlob Sentiment Analysis"):
                    st.metric("Polarity", value=round(textblob_sentiment[0], 2))
                    st.metric("Subjectivity", value=round(textblob_sentiment[1], 2))

            with col2:
                #st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>✨Other Metrics</h3>",
                #            unsafe_allow_html=True)
                st.write("")
                #st.metric(emotion['label'], value=round(emotion['score'], 2))

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

            # --- Display New Features ---
            st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>✨ Additional Analysis</h3>", unsafe_allow_html=True)
            st.subheader("Named Entities")
            st.write(named_entities)

            st.subheader("Part-of-Speech Counts")
            st.write(pos_counts)

            st.subheader("Stopword Density")
            st.write(round(stopword_density, 2))

            st.subheader("Readability Score (Flesch-Kincaid)")
            st.write(round(readability_score, 2))

            st.subheader("Passive Voice Sentences")
            st.write(passive_sentences)

            st.subheader("Pronoun Counts")
            st.write(pronoun_counts)

            st.subheader("First Person vs Third Person")
            st.write(f"First Person: {first_person_count}, Third Person: {third_person_count}")

            st.subheader("Corrected Text (Grammar)")
            st.write(corrected_text)

            st.subheader("Transition Word Count")
            st.write(transition_word_count)

            st.subheader("Text Summary")
            st.write(text_summary)

with tab2:  # File Upload
    st.header("File Upload & Batch Processing")
    uploaded_file = st.file_uploader("Drag & Drop CSV/TXT file here", type=["csv", "txt"], accept_multiple_files=False)

    if uploaded_file:
        st.write("File upload processing has not yet been implemented to incorporate the additional functionality of tab 1.")

with tab3:  # Visualization & Reports
    st.header("Visualization & Reports")
    st.write("Visualization and reporting have not yet been implemented.")
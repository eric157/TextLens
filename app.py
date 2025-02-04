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
import stylecloud

import nltk  # For text preprocessing visualization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import textstat  # For readability scores
import langdetect  # For language detection

import spacy  # For NER and POS tagging

import language_tool_python # Grammar and spelling checks

# --- Page Settings ---
st.set_page_config(
    page_title="TextLens",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Download NLTK Resources (Run once) ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
try:
    wordnet_lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    wordnet_lemmatizer = WordNetLemmatizer()
try:
    word_tokenize("example")
except LookupError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

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

# --- Load spaCy Model ---
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy 'en_core_web_sm' model. This might take a few moments.")
        spacy.cli.download("en_core_web_sm")  # Download if not found
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()

# --- Load LanguageTool ---
@st.cache_resource
def load_language_tool():
    tool = language_tool_python.LanguageTool('en-US')
    return tool

language_tool = load_language_tool()

# --- Helper Functions for New Features ---
@st.cache_data
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    return " ".join(tokens)

@st.cache_data
def get_pos_distribution(text):
    doc = nlp(text)
    pos_counts = {}
    for token in doc:
        pos = token.pos_
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts[pos] = 1
    total = sum(pos_counts.values())
    pos_percentages = {pos: count / total * 100 for pos, count in pos_counts.items()}
    return pos_percentages

@st.cache_data
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

@st.cache_data
def calculate_sentence_complexity_score(text):
    sentences = re.split(r'[.!?]+', text)
    num_sentences = len(sentences) - 1 if sentences else 0
    if num_sentences == 0:
        return 0

    total_length = sum(len(sentence.split()) for sentence in sentences if sentence)
    avg_sentence_length = total_length / num_sentences

    subordinate_conjunctions = ["because", "although", "if", "since", "while", "unless", "until", "when", "where"]
    num_subordinate_clauses = sum(sentence.lower().count(conj) for sentence in sentences for conj in subordinate_conjunctions)
    subordinate_ratio = num_subordinate_clauses / num_sentences if num_sentences > 0 else 0

    return avg_sentence_length + subordinate_ratio

@st.cache_data
def analyze_stopword_density(text):
    tokens = word_tokenize(text.lower())
    total_words = len(tokens)
    if total_words == 0:
        return 0
    stopword_count = len([word for word in tokens if word in stop_words])
    stopword_density = stopword_count / total_words
    return stopword_density

@st.cache_data
def check_grammar_and_spelling(text):
    matches = language_tool.check(text)
    errors = []
    for match in matches:
        if match.replacements:
            errors.append((match.ruleId, match.message, text[match.offset:match.offset + match.errorLength], match.replacements[0])) # ADDING THE OFFENDING WORD
        else:
            errors.append((match.ruleId, match.message, text[match.offset:match.offset + match.errorLength], ""))
    return errors

# --- Functions ---
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
    color = "white" if theme == "light" else "black"  # Stylecloud uses black for dark backgrounds
    try:
        stylecloud.gen_stylecloud(
            text=text[:512],
            background_color=color,
            icon_name='fas fa-cloud',  # Choose an icon
            palette='cartocolors.sequential.Plasma_7',  # Use a palette from cartocolors
            output_name='wordcloud.png'  # Save as a file
        )
        return 'wordcloud.png'  # Return the filename for display
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
         max-width: 200px;
     }}
       /* Style the tab labels */
    .stTabs [data-baseweb="tab-list"] button[role="tab"] {{
        color: {colors['grey_text']}; /* Inactive tab color */
        background-color: {colors['secondary_background']};
        padding: 0.5em 1em; /* Adjust padding as needed */
        border-top-left-radius: 0.5em; /* Rounded corners */
        border-top-right-radius: 0.5em;
        border-bottom: none; /* Remove bottom border */
    }}

    /* Style the active tab label */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: {colors['text_color']}; /* Active tab color */
        background-color: {colors['background_color']};
        border-bottom: none !important; /* Override Streamlit's border */
    }}

    /* Style the tab content area */
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {colors['background_color']};
        padding: 1em;
        border-radius: 0.5em;
        border: 1px solid {colors['secondary_background']};
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

# --- Tabs Configuration ---
tab1, tab2, tab3 = st.tabs(["Text Analysis", "File Upload", "Visualization & Reports"])
theme = DEFAULT_THEME

# --- Central Styling Configuration ---
def apply_styles():
    colors = DARK_MODE if DEFAULT_THEME == "dark" else LIGHT_MODE  # Get current theme colors

    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: 90%;
            padding-top: 5rem;
            padding-right: 5rem;
            padding-left: 5rem;
            padding-bottom: 5rem;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {colors['primary_color']};
        }}
        .stButton>button {{
            color: #4F8BF9;
            border-radius: 10px;
            border: 2px solid #4F8BF9;
            font-weight: bold;
            padding: 0.5em 1em;
        }}
        .stButton>button:hover {{
            background-color: #4F8BF9;
            color: white;
        }}
        .css-1cpxqw2 {{ /* Adjust Checkbox Font */
            font-size: 1rem !important;
        }}
        /* Style the tab labels */
        .stTabs [data-baseweb="tab-list"] button[role="tab"] {{
            color: {colors['grey_text']}; /* Inactive tab color */
            background-color: {colors['secondary_background']};
            padding: 0.5em 1em; /* Adjust padding as needed */
            border-top-left-radius: 0.5em; /* Rounded corners */
            border-top-right-radius: 0.5em;
            border-bottom: none; /* Remove bottom border */
        }}

        /* Style the active tab label */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {colors['text_color']}; /* Active tab color */
            background-color: {colors['background_color']};
            border-bottom: none !important; /* Override Streamlit's border */
        }}

        /* Style the tab content area */
        .stTabs [data-baseweb="tab-panel"] {{
            background-color: {colors['background_color']};
            padding: 1em;
            border-radius: 0.5em;
            border: 1px solid {colors['secondary_background']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_styles() #apply styling
# --- Text Analysis Tab ---
with tab1:  # Text Analysis
    st.header("Text Analysis Dashboard", divider="rainbow")  # Modern header

    st.markdown(f"Enter text for analysis (Maximum: {MAX_WORDS} words):", unsafe_allow_html=True)
    text_input = st.text_area("Enter text for analysis:", height=150, key="text_area", max_chars=MAX_WORDS * 7)
    word_count = len(text_input.split())
    st.markdown(f'<div id="word-count">Word count: {word_count}</div>', unsafe_allow_html=True)  # show words limit

    # --- Feature Toggles ---
    st.subheader("Analysis Options", divider="blue")
    col1, col2 = st.columns(2)

    with col1:
        show_preprocessing = st.checkbox("Show Preprocessed Text")
        show_readability = st.checkbox("Show Readability Scores")
        detect_language = st.checkbox("Detect Language")
        show_pos_distribution = st.checkbox("Show POS Tag Distribution")

    with col2:
        show_ner = st.checkbox("Show Named Entity Recognition")
        show_complexity = st.checkbox("Show Sentence Complexity")
        show_stopword_density = st.checkbox("Show Stopword Density")
        show_grammar_errors = st.checkbox("Check Grammar and Spelling")

    if text_input:
        with st.spinner('Analyzing Text...'):
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

            # --- Metrics Display ---
            st.subheader("Key Metrics", divider="green")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Sentiment", value=sentiment_result['label'], delta=sentiment_result['score'])
            with col2:
                st.metric("Reading Time (mins)", reading_time)
            with col3:
                st.metric("Lexical Diversity", round(lexical_diversity, 2))
            with col4:
                st.metric("Sentence Count", sentence_count)

            col5, col6 = st.columns(2)
            with col5:
                with st.expander("TextBlob Sentiment Analysis"):
                    st.metric("Polarity", value=round(textblob_sentiment[0], 2))
                    st.metric("Subjectivity", value=round(textblob_sentiment[1], 2))
            with col6:
                 st.metric("Avg. Word Length", round(avg_word_length, 2)) # adding it to the UI on request

            st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>🔑 Keywords</h3>", unsafe_allow_html=True)
            st.write(", ".join(keywords))
            if hashtags:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>#️⃣ Hashtags</h3>", unsafe_allow_html=True)
                st.write(", ".join(hashtags))
            if emojis:
                st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'>😀 Emojis</h3>", unsafe_allow_html=True)
                st.write(" ".join(emojis))

            # --- Implemented Features ---
            st.subheader("Detailed Analysis", divider="violet")
            if show_preprocessing:
                st.subheader("Text Preprocessing")
                preprocessed_text = preprocess_text(text_input)
                st.write(preprocessed_text)

            if show_readability:
                st.subheader("Readability Scores")
                flesch_kincaid = textstat.flesch_kincaid_grade(text_input)
                st.metric("Flesch-Kincaid Grade Level", flesch_kincaid)

            if detect_language:
                st.subheader("Language Detection")
                try:
                    language = langdetect.detect(text_input)
                    st.write(f"Detected Language: {language}")
                except langdetect.LangDetectException:
                    st.warning("Could not detect language.")

            if show_pos_distribution:
                st.subheader("Part-of-Speech Tag Distribution")
                pos_counts = get_pos_distribution(text_input)
                st.write(pos_counts)

            if show_ner:
                st.subheader("Named Entity Recognition")
                entities = extract_named_entities(text_input)
                st.write(entities)

            if show_complexity:
                st.subheader("Sentence Complexity Score")
                complexity_score = calculate_sentence_complexity_score(text_input)
                st.metric("Complexity Score", round(complexity_score, 2))

            if show_stopword_density:
                st.subheader("Stopword Density")
                stopword_density = analyze_stopword_density(text_input)
                st.metric("Stopword Density", round(stopword_density, 2))

            if show_grammar_errors:
                st.subheader("Grammar and Spelling Check")
                grammar_errors = check_grammar_and_spelling(text_input)
                if grammar_errors:
                    st.warning("Potential Grammar and Spelling Errors Found!")
                    for error_id, error_message, error_word, suggestion in grammar_errors:
                        st.write(f"- **{error_message}** (Error: '{error_word}', Suggestion: '{suggestion}')")
                else:
                    st.success("No grammar or spelling errors found.")

            st.markdown(f"<h3 style='color:{DARK_MODE['primary_color']} ;'> ☁️ Word Cloud Visualization</h3>", unsafe_allow_html=True)
            wordcloud_file = generate_wordcloud(text_input, theme=theme)
            if wordcloud_file:  # Check if the file was successfully created
                st.image(wordcloud_file)

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
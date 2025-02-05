import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import re
import emoji
from textblob import TextBlob
import plotly.express as px
from streamlit_extras.app_logo import add_logo
from streamlit_option_menu import option_menu
from PIL import Image

# --- CONSTANTS ---
MAX_WORDS = 300  # setting max value of inputs
DEFAULT_THEME = "dark"
SENTIMENT_MODEL = "sentiment-analysis"
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
KEYWORD_MODEL = "google/flan-t5-base"

# --- Theme Colors ---
LIGHT_MODE = {
    "primary_color": "#1E90FF",
    "background_color": "#ffffff",
    "text_color": "#000000",
    "secondary_background": "#f5f5f5",
    "grey_text": "#454545",
}
DARK_MODE = {
    "primary_color": "#1E90FF",
    "background_color": "#0E1117",
    "text_color": "#ffffff",
    "secondary_background": "#181818",
    "grey_text": "#919191",
}

# --- Page Configuration ---
st.set_page_config(
    page_title="TextLens",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
def inject_custom_css(theme):
    colors = DARK_MODE if theme == "dark" else LIGHT_MODE
    st.markdown(
        """
        <style>
        /* General Styles */
        body {
            color: """ + colors['text_color'] + """;
            background-color: """ + colors['background_color'] + """;
        }
        .stApp {
           background-color: """ + colors['background_color'] + """;
        }
        .block-container {
           background-color: """ + colors['background_color'] + """;
            color: """ + colors['text_color'] + """;
            padding: 1rem 2rem;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: """ + colors['primary_color'] + """;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        h1 {
            font-size: 2.5em;
            text-align: center;
        }
        h2 {
            font-size: 2em;
        }
        h3 {
            font-size: 1.6em;
        }
        .css-10tr677, .css-1x63k98, .css-1v0mbdr {
            color: """ + colors['text_color'] + """; /* Consistent text color for various Streamlit elements */
        }

        /* Sidebar */
        .css-1adrfps { /* Adjusts the Sidebar Size  */
           max-width: 280px !important; /* Set max width for bigger sidebars  */
         }
        [data-testid="stSidebar"] {
           background-color: """ + colors['secondary_background'] + """;
          color: """ + colors['text_color'] + """;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: """ + colors['primary_color'] + """; /* Titles in primary Color  */
        }
        /* Input Elements */
         [data-baseweb="base-input"], textarea[data-streamlit="text-area"] {
              background-color: """ + colors['secondary_background'] + """;
              color: """ + colors['text_color'] + """ !important;
               border-radius: 0.3rem;
               padding: 0.5rem 0.75rem; /* Padding on Input elements  */
         }

       textarea[data-streamlit="text-area"]::placeholder {
        color: """ + colors['grey_text'] + """ !important; /* Placeholder Color  */
             opacity: 0.8;
           }

        /* Buttons */
         .stButton>button {
                color: """ + colors['text_color'] + """;
                 background-color: """ + colors['primary_color'] + """;
                 font-weight: 500;
                  border: none;
                  border-radius: 0.3rem;
                  padding: 0.5rem 1rem;
            }

        .stDownloadButton>button {  /* style Downloads */
          background-color: """ + colors['primary_color'] + """;
           border: none;
          color: """ + colors['text_color'] + """;
            }

       /* Tables,Metrics and DataFrames styles here   */
         .stDataFrame  {
               border: 1px solid """ + colors['primary_color'] + """
             }
         .stMetricLabel, .stMetricDelta {  /* Style Metrics texts labels and delta   */
                color: """ + colors['text_color'] + """;
                 opacity:0.8
               }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Model Loading Functions ---
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(SENTIMENT_MODEL)

@st.cache_resource
def load_emotion_pipeline():
    return pipeline("text-classification", model=EMOTION_MODEL)

@st.cache_resource
def load_keyword_pipeline():
    return pipeline("text2text-generation", model=KEYWORD_MODEL)

# --- Text Analysis Functions ---
def analyze_sentiment(text):
    """Analyzes the sentiment of the given text."""
    try:
        sentiment_result = load_sentiment_pipeline()(text[:512])
        return sentiment_result[0]
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return {"label": "Error", "score": 0.0}

def analyze_emotions(text):
    """Analyzes the emotions expressed in the given text."""
    try:
        emotions_result = load_emotion_pipeline()(text[:512])
        return emotions_result
    except Exception as e:
        st.error(f"Error during emotion analysis: {e}")
        return [{"label": "Error", "score": 0.0}]

def extract_keywords(text):
    """Extracts keywords from the given text."""
    try:
        keyword_pipeline = load_keyword_pipeline()
        prompt = f"Extract keywords: {text}"
        keywords_result = keyword_pipeline(prompt[:512], max_length=50, num_return_sequences=1)
        return [res['generated_text'] for res in keywords_result]
    except Exception as e:
        st.error(f"Error during keyword extraction: {e}")
        return []

def analyze_hashtags(text):
    """Analyzes hashtags found in the given text."""
    try:
        hashtags = re.findall(r"#(\w+)", text[:512])
        return hashtags
    except Exception as e:
        st.error(f"Error during hashtag analysis: {e}")
        return []

def analyze_emojis(text):
    """Analyzes emojis used in the given text."""
    try:
        emojis = [char for char in text if char in emoji.EMOJI_DATA]
        return emojis
    except Exception as e:
        st.error(f"Error during emoji analysis: {e}")
        return []

def analyze_textblob_sentiment(text):
    """Analyzes the sentiment of the given text using TextBlob."""
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    except Exception as e:
        st.error(f"Error during TextBlob sentiment analysis: {e}")
        return 0.0, 0.0

# --- Text Statistics Functions ---
def estimate_reading_time(text):
    """Estimates the reading time of the given text in minutes."""
    words = len(text.split())
    reading_speed_wpm = 200  # Average reading speed in words per minute
    reading_time = words / reading_speed_wpm
    return round(reading_time, 2)

def calculate_lexical_diversity(text):
    """Calculates the lexical diversity of the given text."""
    words = text.split()
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def count_sentences(text):
    """Counts the number of sentences in the given text."""
    sentences = re.split(r'[.!?]+', text)  # Splitting by common sentence delimiters
    return len(sentences) - 1 if sentences else 0  # Adjust count to exclude empty strings

def analyze_average_word_length(text):
    """Calculates the average word length of the given text."""
    words = text.split()
    if not words:
        return 0.0
    total_length = sum(len(word) for word in words)
    return total_length / len(words)

# --- Visualization Functions ---
def generate_wordcloud(text, theme="light"):
    """Generates a word cloud from the given text."""
    color = "white" if theme == "light" else "#181818"
    try:
        wordcloud = WordCloud(width=800, height=400, background_color=color, colormap='viridis').generate(text[:512])  # Use viridis colormap
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        return plt
    except Exception as e:
        st.error(f"Error during wordcloud generation: {e}")
        return None

# --- Utility Functions ---
def display_download_button(df, file_format, filename):
    """Displays a download button for the given DataFrame."""
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

# --- Sidebar Content ---
def create_sidebar():
    """Creates the sidebar content."""
    logo_path = "logo.png"

    with st.sidebar:
        # Load logo
        try:
            with Image.open(logo_path) as logo:
                st.sidebar.image(logo, width=150, use_container_width=True)
                st.markdown(""" <style> [data-testid='stImage'] > div > img {  border-radius: 10px}</style>""",
                            unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Error Loading Logo: {e}", icon="🚨")

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

# --- Main App Content ---
def main():
    """Main function to run the Streamlit app."""
    inject_custom_css(DEFAULT_THEME)  # Inject custom CSS based on the theme
    create_sidebar()

    # Title
    st.markdown("<h1 style='text-align: center;'>TextLens – Analyze text with clarity</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Text Analysis", "File Upload", "Visualization & Reports"])
    theme = DEFAULT_THEME

    with tab1:  # Text Analysis
        text_analysis_tab(theme)

    with tab2:  # File Upload
        file_upload_tab()

    with tab3:  # Visualization & Reports
        visualization_tab()

# --- Tab Functions ---
def text_analysis_tab(theme):
    """Content for the Text Analysis tab."""
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
                st.markdown("<h3>📊 Sentiment Analysis</h3>", unsafe_allow_html=True)
                if sentiment_result['label'] == "Error":
                    st.error("Sentiment Analysis Failed")
                else:
                    st.metric("Sentiment", value=sentiment_result['label'], delta=sentiment_result['score'])

                with st.expander("📌 TextBlob Sentiment Analysis"):
                    st.metric("Polarity", value=round(textblob_sentiment[0], 2))
                    st.metric("Subjectivity", value=round(textblob_sentiment[1], 2))

            with col2:
                st.markdown("<h3>💖 Emotion Classification</h3>", unsafe_allow_html=True)
                for emotion in emotion_result:
                    st.metric(emotion['label'], value=round(emotion['score'], 2))

            st.markdown("<h3>🔑 Keyword Extraction</h3>", unsafe_allow_html=True)
            st.write(", ".join(keywords))

            if hashtags:
                st.markdown("<h3>#️⃣ Hashtags</h3>", unsafe_allow_html=True)
                st.write(", ".join(hashtags))

            if emojis:
                st.markdown("<h3>😀 Emojis</h3>", unsafe_allow_html=True)
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

            st.markdown("<h3> ☁️ Word Cloud Visualization</h3>", unsafe_allow_html=True)
            wordcloud_fig = generate_wordcloud(text_input, theme=theme)
            if wordcloud_fig:  # Check if the figure was successfully created
                st.pyplot(wordcloud_fig)

def file_upload_tab():
    """Content for the File Upload tab."""
    st.header("File Upload & Batch Processing")
    if st.button("Clear Analysis"):
        if 'df' in st.session_state:
            del st.session_state.df
    uploaded_files = st.file_uploader("Drag & Drop CSV/TXT files here", type=["csv", "txt"], accept_multiple_files=True)

    if uploaded_files:
        all_dfs = []

        for uploaded_file in uploaded_files:
            with st.spinner(f'Processing the file: {uploaded_file.name}...'):
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                        if 'text' not in df.columns:
                            st.error(f"CSV file '{uploaded_file.name}' must have a 'text' column.", icon="🚨")
                            continue
                        texts = df['text'].tolist()

                    elif uploaded_file.name.endswith(".txt"):
                        text_file = uploaded_file.read().decode('utf-8')
                        texts = [text_file]
                        df = pd.DataFrame({'text': texts})

                    else:
                        st.error(f"Unsupported file type: '{uploaded_file.name}'. Please upload a CSV or TXT file.", icon="🚨")
                        continue
                        
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

                    all_dfs.append(df)

                except Exception as e:
                    st.error(f"An error occurred during file processing: {uploaded_file.name}: {e}", icon="🚨")

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            st.subheader("Analysis Results")
            st.dataframe(final_df, height=300)
            st.session_state.df = final_df
        else:
            st.warning("No files were successfully processed.")

def visualization_tab():
    """Content for the Visualization & Reports tab."""
    if 'df' in st.session_state:
        df = st.session_state.df
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
                               color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig_sentiment)

        # Emotion Distribution Bar Chart
        emotion_counts = df['emotion'].value_counts()
        fig_emotion = px.bar(emotion_counts, x=emotion_counts.index, y=emotion_counts.values,
                             title="Emotion Distribution",
                             color_discrete_sequence=px.colors.sequential.Viridis)
        fig_emotion.update_layout(xaxis_title="Emotion", yaxis_title="Count")
        st.plotly_chart(fig_emotion)

        # Reading Time Distribution Histogram
        fig_reading_time = px.histogram(df, x="reading_time", title="Reading Time Distribution",
                                        color_discrete_sequence=px.colors.sequential.Cividis)
        fig_reading_time.update_layout(xaxis_title="Reading Time (minutes)", yaxis_title="Frequency")
        st.plotly_chart(fig_reading_time)
    else:
        st.info("Upload a file to view visualizations.")

if __name__ == "__main__":
    main()
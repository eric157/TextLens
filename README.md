# 📜 TextLens: Emotion and Sentiment Analysis Web App

![TextLens](https://textlens.streamlit.app/)

## 🚀 Overview

**TextLens** is an advanced **Natural Language Processing (NLP)** web application built with **Streamlit**. It utilizes state-of-the-art **transformer models** from **Hugging Face** for **emotion detection** and **text generation**. This tool allows users to analyze the emotions embedded in text and generate relevant responses, making it useful for applications like sentiment analysis, chatbot enhancement, and content moderation.

## 🎯 Features

- 🔍 **Emotion Analysis**: Uses a fine-tuned **RoBERTa** model to classify emotions in the given text.
- 📝 **Text Generation**: Leverages **Google’s FLAN-T5** model to generate relevant responses based on the detected emotions.
- 🚀 **Fast and Efficient**: Powered by **Hugging Face Pipelines**, making it seamless and optimized.
- 🌐 **Deployed Online**: Access the app at **[TextLens](https://textlens.streamlit.app/)**.

---

## 🏗️ Tech Stack & Models Used

### 1️⃣ **Frontend: Streamlit**
- Built using **Streamlit**, a powerful Python framework for creating interactive web applications.
- Provides a **user-friendly UI** with minimalistic design.
- Supports **real-time interaction** with the models.

### 2️⃣ **Backend: Hugging Face Transformers**
- Uses **Hugging Face Pipelines** to handle inference efficiently.
- Two key transformer models are used:
  
  #### **a) Emotion Detection: `SamLowe/roberta-base-go_emotions`**
  - **Architecture**: RoBERTa (Robustly optimized BERT)
  - **Dataset**: Fine-tuned on the **GoEmotions dataset** (Google’s dataset containing 27 labeled emotions).
  - **Function**: Predicts the **primary emotion** in the text.
  - **Why RoBERTa?**:
    - Pre-trained using dynamic masking, improving performance over standard BERT.
    - Optimized for **contextual emotion understanding**.

  #### **b) Text Generation: `google/flan-t5-base`**
  - **Architecture**: T5 (Text-to-Text Transfer Transformer) 
  - **Variant**: FLAN-T5 (Fine-tuned for better reasoning & generalization)
  - **Function**: Generates text conditioned on detected emotions.
  - **Why FLAN-T5?**:
    - Uses **instruction tuning** for better zero-shot performance.
    - Handles a variety of NLP tasks efficiently.

### 3️⃣ **Infrastructure**
- **Hugging Face’s `transformers` library** for model inference.
- **Torch (PyTorch)** for efficient tensor operations.
- **Streamlit hosting** for seamless deployment.

---

## 🔧 Installation & Usage

### 📥 Prerequisites
Ensure you have **Python 3.8+** and install the required libraries:

```bash
pip install streamlit transformers torch
```

### 🚀 Running the App Locally
Clone the repository:

```bash
git clone https://github.com/yourusername/TextLens.git
cd TextLens
```

Run the application:

```bash
streamlit run app.py
```

---

## 🛠️ How It Works (Technical Breakdown)

1. **User Input**:  
   - The user enters a sentence in the text box.

2. **Emotion Analysis (RoBERTa Model)**:  
   - The input is passed through `SamLowe/roberta-base-go_emotions`.
   - The model outputs a probability distribution across 27 possible emotions.
   - The **highest probability** emotion is selected.

3. **Text Generation (FLAN-T5 Model)**:  
   - The selected emotion is used as a **prompt** for `google/flan-t5-base`.
   - The model generates a response based on the detected emotion.

4. **Display the Results**:  
   - The original input, detected emotion, and AI-generated text are shown on the UI.

---

## 🚀 Future Enhancements

🔮 **Advanced Multi-Emotion Classification**  
- Instead of selecting just **one dominant emotion**, a multi-label classification approach can be implemented.

🧠 **Fine-Tuned Text Generation**  
- Custom fine-tuning of **FLAN-T5** on emotion-labeled response datasets.

⚡ **Performance Optimization**  
- Implement **ONNX Runtime** or **TensorRT** for faster inference.

📡 **API-Based Access**  
- Provide a REST API using **FastAPI** for external integrations.

🤖 **Integration with Chatbots**  
- Seamless plug-in for chatbot applications like **Rasa** or **Dialogflow**.

---

## 📌 Deployment
The app is deployed at **[TextLens](https://textlens.streamlit.app/)** and hosted using **Streamlit Cloud**.
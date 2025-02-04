# 📜 TextLens: Emotion and Sentiment Analysis Web App  

![Logo](logo.png)  

🌍 **Live Demo**: [TextLens](https://textlens.streamlit.app/)  

## 🚀 Overview  

**TextLens** is an advanced **Natural Language Processing (NLP)** web application built with **Streamlit**. It leverages **transformer models** from **Hugging Face** for **emotion detection** and **text generation**. This tool is ideal for applications in **sentiment analysis, chatbot enhancement, and content moderation**.  

## 🎯 Features  

- 🔍 **Emotion Analysis**: Uses a **fine-tuned RoBERTa model** to classify emotions in text.  
- 📝 **Text Generation**: Utilizes **Google’s FLAN-T5** model to generate responses based on detected emotions.  
- ⚡ **Real-time Processing**: Powered by **Hugging Face Pipelines** for **fast inference**.  
- 🌐 **Web App**: Accessible at **[TextLens](https://textlens.streamlit.app/)**.  

---

## 🏗️ Tech Stack & Models Used  

### 1️⃣ **Frontend: Streamlit**  
- Built using **Streamlit**, a lightweight Python framework for creating **interactive web applications**.  
- Provides a **user-friendly UI** with real-time interactions.  

### 2️⃣ **Backend: Hugging Face Transformers**  
Two powerful transformer models are used:  

#### **a) Emotion Detection: `SamLowe/roberta-base-go_emotions`**  
- **Model Type**: RoBERTa (Robustly Optimized BERT Pretraining Approach).  
- **Dataset**: Fine-tuned on **Google’s GoEmotions dataset** (27 emotion labels).  
- **Function**: Detects emotions in text with high accuracy.  
- **Why RoBERTa?**  
  - Pre-trained using **dynamic masking**, improving **contextual emotion understanding**.  
  - Outperforms traditional **BERT** on **emotion classification tasks**.  

#### **b) Text Generation: `google/flan-t5-base`**  
- **Model Type**: T5 (Text-to-Text Transfer Transformer).  
- **Variant**: **FLAN-T5**, optimized for **zero-shot reasoning and text generation**.  
- **Function**: Generates a **text response** based on detected emotion.  
- **Why FLAN-T5?**  
  - Uses **instruction tuning** for better generalization.  
  - Handles multiple NLP tasks efficiently without additional fine-tuning.  

### 3️⃣ **Infrastructure**  
- **Hugging Face’s `transformers` library** for NLP tasks.  
- **Torch (PyTorch)** for deep learning model inference.  
- **Streamlit Cloud** for seamless deployment.  

---

## 🔧 Installation & Usage  

### 📥 Prerequisites  
Ensure you have **Python 3.8+** installed. Install dependencies using:  

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

1️⃣ **User Input**  
- The user enters text into the input box.  

2️⃣ **Emotion Analysis (`SamLowe/roberta-base-go_emotions`)**  
- The model predicts a **probability distribution** over 27 emotions.  
- The emotion with the **highest confidence score** is selected.  

3️⃣ **Text Generation (`google/flan-t5-base`)**  
- The detected emotion is used as a **prompt** for `FLAN-T5`.  
- The model generates a **contextually relevant** response.  

4️⃣ **Display the Results**  
- The **original text, detected emotion, and AI-generated response** are displayed in the UI.  

---

## 🚀 Future Enhancements  

🔮 **Multi-Emotion Classification**  
- Implement **multi-label classification** to detect **multiple emotions** in a single text.  

🧠 **Fine-Tuned Text Generation**  
- Fine-tune **FLAN-T5** on an **emotion-response dataset** for improved response generation.  

⚡ **Performance Optimization**  
- Deploy models using **ONNX Runtime** or **TensorRT** for **faster inference**.  

📡 **API-Based Access**  
- Develop a **REST API** using **FastAPI** for external integrations.  

🤖 **Chatbot Integration**  
- Enhance chatbot interactions using **Rasa** or **Dialogflow**.  

---

## 📌 Deployment  
The app is **live and hosted** on **Streamlit Cloud** at:  

🔗 **[TextLens](https://textlens.streamlit.app/)**  

---
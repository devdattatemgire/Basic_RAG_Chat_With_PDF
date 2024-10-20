# 📝 Chat with Tamil PDF: A RAG Implementation Using Gemini API and LangChain

Welcome to **Chat with Tamil PDF**! This is a basic implementation of **Retrieval-Augmented Generation (RAG)** using the **Gemini API** and **LangChain**. The app lets you upload a PDF (particularly Tamil PDFs), split the content into manageable chunks, and interact with it through an intelligent chatbot powered by **Google Gemini LLM** and **FAISS indexing**. 🧠📄

## Features:
- **Upload a PDF**: Easily upload your PDF and start chatting with its contents. 📤
- **Text Chunking**: The content is split into manageable chunks using LangChain's recursive text splitter for better context retention. 🔗
- **FAISS Indexing**: The app dynamically creates and stores FAISS indexes to improve query retrieval. 🗂️
- **Google Gemini LLM**: Harness the power of Google's Generative AI for highly accurate, retrieval-based answers. 💬🤖

## Setup Instructions:

### 1. Clone the Repository:
To clone this repository, run the following command in your terminal:
```
git clone https://github.com/devdattatemgire/Basic_RAG_Chat_With_PDF.git
```

### 2. Create a Virtual Environment:
To create a virtual environment, use the following command:
```
python -m venv .venv
```

### 3. Activate Virtual Environment:
  - On Windows
    ```
    .\.venv\Scripts\activate
    ```
  - On Linux
    ```
    source .venv/bin/activate
    ```
### 4. Install Required Packages:
After activating the virtual environment, install the required packages using:
```
pip install -r requirements.txt
```

### 5. Create a .env File:
Create a file named .env in the root of your project directory and add your Gemini API key:
```
GOOGLE_API_KEY=<your_key>
```

### 6. Run the Streamlit App:
Start the Streamlit app by running the following command:
```
streamlit run app.py
```

## 🎉 Happy coding! Remember: Code is like humor. When you have to explain it, it’s bad! 😂
  

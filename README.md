# 🧠 AnalytiBot – AI-Powered Research Analysis Assistant

AnalytiBot is an intelligent research assistant that enables **natural-language interaction with unstructured research data** such as **PDFs, web articles (URLs), CSV datasets, and images**.

It combines **Large Language Models (LLMs)**, **LangChain**, **OpenAI embeddings**, and **FAISS vector search** to deliver accurate, source-aware answers and advanced statistical insights through a **Streamlit-based web interface**.

This project was developed as part of a **Bachelor of Technology (B.Tech) thesis** in **Computer Science and Engineering (Cyber Physical Systems)**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Use Cases](#-use-cases)
- [Future Scope](#-future-scope)
- [Thesis Reference](#-thesis-reference)
- [License](#-license)

---

## 🔍 Overview

With the rapid growth of academic publications, manually analyzing research articles has become time-consuming and error-prone.

AnalytiBot addresses this challenge by providing a **Retrieval-Augmented Generation (RAG)** based chatbot that allows researchers to:
- Query large documents in natural language
- Retrieve answers with source references
- Perform statistical analysis on datasets
- Retain all prompts and responses within a session

---

## 🚀 Key Features

### 🔹 AI-Driven Research Q&A
- Natural language question answering
- Source-aware responses
- Session-level question & answer retention

### 🔹 Multi-Source Data Ingestion
- 📄 PDFs (research papers, reports)
- 🌐 URLs (news articles, journals)
- 📊 CSV files (datasets)
- 🖼️ Images → OCR → searchable text

### 🔹 Vector Search & Retrieval (RAG)
- Recursive text chunking using LangChain
- Semantic embeddings using OpenAI
- High-performance similarity search with FAISS
- Pickle-based persistent vector storage

### 🔹 Statistical & Data Analysis
- Linear Regression
- Multiple Regression
- T-Tests (One-sample, Paired, Independent)
- ANOVA (One-way, Two-way)
- Z-Test
- Correlation Analysis (Pearson & Spearman)
- Chi-Square Tests
- Time-Series Analysis:
  - Moving Average
  - Exponential Smoothing
  - Autocorrelation
  - Seasonality
- Visualizations:
  - Histograms
  - Box Plots
  - Q-Q Plots
  - Heatmaps

### 🔹 User Experience
- Streamlit-based web UI
- Voice input support
- Response time tracking
- Session history retention

---

## 🧩 System Architecture

AnalytiBot follows a **Retrieval-Augmented Generation (RAG)** workflow:

1. Data ingestion (PDF / URL / CSV / Image)
2. Recursive text splitting
3. Vector embedding using OpenAI
4. Storage in FAISS vector database
5. Semantic similarity search
6. LLM-based answer generation
7. Session-level prompt & answer retention

---

## 🛠️ Tech Stack

| Category | Technologies |
|--------|-------------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-3.5-Turbo / GPT-3.5-Turbo-16k |
| Framework | LangChain |
| Vector DB | FAISS |
| Embeddings | OpenAI Embeddings |
| OCR | EasyOCR |
| Speech Input | SpeechRecognition |
| Data Analysis | Pandas, StatsModels, SciPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Backend | Python 3.9 |

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/AnalytiBot.git
cd AnalytiBot
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Configure Environment Variables
Create a .env file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
```
### 4️⃣ Run the Application
```bash
streamlit run main.py
```

## 🧪 Usage Guide

1. Launch the Streamlit application
2. Select the data input type:
   - URL
   - PDF
   - CSV
   - Image
3. Upload data or enter URLs
4. Ask questions in natural language
5. View answers, sources, and retained session history
6. Perform statistical analysis for CSV datasets


## 🎥 Demo

▶️ **Demo Video (MP4):**  
[Click here to watch the demo of how to run the project](assets/demo1.mp4)

[Click here to watch the demo of how to upload and search using URLs](assets/demo2.mp4)

[Click here to watch the demo of how to upload csv files and do statistical analysis](assets/demo3.mp4)

[Click here to watch the demo of how to generally ask questions on the UI](assets/demo4.mp4)

# 📧 Email & SMS Spam Detection System

This repository contains a **Spam Detection System** that classifies messages (SMS or emails) as **Spam** or **Not Spam (Ham)**. It uses **Natural Language Processing (NLP)** techniques and **machine learning models** to detect spam messages efficiently.  

---

## 🚀 Live Demo

- 🌐 [Web App (Render Deployment)](https://spam-detection-system-2.onrender.com)  
- 🌐 [Web App (Streamlit Deployment)](https://spam-detection-system-st.streamlit.app/)

---

## 📑 Table of Contents  

- [Introduction](#-introduction)  
- [Why We Need This Project](#-why-we-need-this-project)  
- [Project Flow](#-project-flow)  
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)  
- [Model Building & Evaluation](#-model-building--evaluation)  
- [Web App & Deployment](#-web-app--deployment)  
- [Tech Stack](#-tech-stack)  
- [Summary](#-summary)  

---

## 📘 Introduction  

Spam messages can be annoying or dangerous (phishing, scams, fraud). This project implements a **Spam Detection System** to classify messages and provide insights into spam patterns.  

It includes a **Streamlit web app** that allows users to:  
- Predict single messages  
- Upload CSV files for batch predictions  
- Visualize spam vs ham statistics and word clouds  

---

## ❓ Why We Need This Project  

- Protect users from unwanted or malicious messages.  
- Analyze message content to detect spam patterns.  
- Gain hands-on experience with **NLP and machine learning**.  
- Learn full project workflow: **data → preprocessing → modeling → frontend → deployment**.  

---

## 🚀 Project Flow  

1. **Data Cleaning** → Handle missing/duplicate data and encode labels.  
2. **Exploratory Data Analysis (EDA)** → Analyze message characteristics and spam/ham distribution.  
3. **Text Preprocessing** → Tokenization, removing stopwords, stemming, TF-IDF vectorization.  
4. **Model Building** → Train classifiers (Multinomial Naive Bayes).  
5. **Model Evaluation** → Accuracy, confusion matrix, precision.  
6. **Improvement** → Tune preprocessing, try alternative models, handle class imbalance.  
7. **Web App** → Build an interactive Streamlit interface.  
8. **Deployment** → Deploy locally or on Streamlit Cloud/Heroku.  

---

## 📂 Data Cleaning & Preprocessing  

### Step 1: Data Cleaning
- Drop unnecessary columns  
- Rename columns: `v1` → `target`, `v2` → `text`  
- Encode target labels using LabelEncoder  
- Check for:
  - Missing values
  - Duplicate rows  

### Step 2: Exploratory Data Analysis (EDA)
- Calculate spam vs ham percentage  
- Analyze message statistics:
  - Number of characters
  - Number of words
  - Number of sentences  
- Use **NLTK** for tokenization:
```python
import nltk
nltk.download('punkt')
```  

### Step 3: Text Preprocessing
- Convert text to lowercase  
- Tokenize messages  
- Remove special characters  
- Remove stopwords and punctuation  
- Apply stemming (**PorterStemmer**)  
- Created a function to perform all preprocessing steps  
- Generated **word clouds** for both spam and ham messages  

---

## 🏗 Model Building & Evaluation  

- Started with **Naive Bayes classifiers**  
- Selected **TF-IDF + Multinomial Naive Bayes** for best performance  
- Evaluated using:
  - Accuracy  
  - Confusion Matrix  
  - Precision  

- Future improvements:
  - Try ensemble models (Random Forest, XGBoost)  
  - Hyperparameter tuning  
  - Handle imbalanced data with SMOTE or class weights  

---

## 💻 Web App & Deployment  

- Built using **Streamlit**  
- Features implemented:
  - Single message prediction  
  - Sample messages for testing  
  - CSV upload for batch prediction  
  - Metrics cards: Total messages, Spam, Ham  
  - Most common spam keywords  
  - Visualizations:
    - Spam vs Ham pie chart  
    - Word cloud of spam messages  
- Deployment options: **Streamlit Cloud**, **Heroku**  

---

## ⚙️ Tech Stack  

- **Language:** Python 🐍  
- **Libraries:** Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, WordCloud, Streamlit  
- **Model Persistence:** Pickle / Joblib  
- **Frontend:** Streamlit  
- **Deployment:** Streamlit Cloud / Heroku  

---

## 📊 Summary  

This project builds an **Email/SMS Spam Detection System** using **NLP and machine learning**.  

**Project workflow:**  
**Data → Cleaning → EDA → Text Preprocessing → Model Building → Evaluation → Web App → Deployment**  

It demonstrates real-world application of **text classification** and provides a **user-friendly web interface** for spam detection.

---


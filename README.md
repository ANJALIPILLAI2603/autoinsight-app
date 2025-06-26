

# 🧠 AutoInsight – AI-Powered Dataset Summary App

AutoInsight is a smart data analysis tool that reads a dataset and instantly generates a natural language summary using Google’s Gemini 2.5 Flash model.

No manual EDA. No technical jargon. Just clean, readable insights — fast.

---
## 🎯 Who This App Is For

This app is designed for:

- 💼 **Business analysts** who want instant summaries without coding
- 🧪 **Data scientists** who need quick EDA before modeling
- 📊 **Non-technical stakeholders** who need human-readable reports
- 🧑‍🏫 **Students or educators** demonstrating NLP/NLG in data analytics

Whether you're in banking, retail, healthcare, or ed-tech — AutoInsight helps **make raw data talk**.

---

## 🚀 What It Does

- 📥 Upload any CSV file (like churn, sales, customer data)
- 🔍 Auto-detects numeric and categorical columns
- 📊 Calculates key stats (mean, std dev, max, modes)
- 🧠 Uses Gemini 2.5 Flash to write a human-friendly summary
- 📄 Outputs a clean AI report (can be exported as PDF)

---

## ⚙️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key to a .env file
echo GEMINI_API_KEY=your-api-key > .env

# 3. Run the app
streamlit run app.py
````

---

## 🔐 API Key Setup

1. Get your key from [Makersuite](https://makersuite.google.com/app/apikey)
2. Create a `.env` file:

   ```env
   GEMINI_API_KEY=your-key-here
   ```
3. Never commit your `.env` file (already ignored via `.gitignore`)

---

## 📁 Files Overview

| File               | Purpose                 |
| ------------------ | ----------------------- |
| `app.py`           | Streamlit UI            |
| `ai_summary.py`    | Gemini + NLP logic      |
| `.env`             | Stores API key (secure) |
| `requirements.txt` | Dependencies            |
| `README.md`        | You’re reading it 😊    |

---


---

## 🧑‍💻 Made By

**Anjali Pillai**
Data Science • AI • ML • NLP
[LinkedIn] - https://www.linkedin.com/in/anjali-pillai-367b2b259/

---

## 📜 License

MIT



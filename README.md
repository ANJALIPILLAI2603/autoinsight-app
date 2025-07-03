

# 📊 AutoInsight: Smart Report & ML Predictor

AutoInsight is a powerful Streamlit app that:
- Automatically generates insightful visualizations and AI summaries from any uploaded CSV file.
- Trains a Machine Learning model (Regression or Classification) with minimal effort.
- Allows users to make predictions on new input data.
- Generates a downloadable PDF report containing visualizations and AI summaries.

🚀 Click the link above to try the app instantly (hosted on Streamlit Cloud).
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autoinsight-app.streamlit.app/)

---
## 🎯 Who is this App For?

- **Business Analysts** – Quickly analyze customer/sales/financial data with auto-insights.
- **Data Scientists & ML Engineers** – Save time on initial EDA, modeling, and report generation.
- **Students & Educators** – Use this tool to demonstrate end-to-end data analysis and ML in minutes.
- **Startup Founders & Product Managers** – Make data-driven decisions with no code.
- **Researchers** – Understand patterns in research data, generate summaries and visuals instantly.

---

## 💼 Business Value & Use Cases

- 🔍 **Accelerated Decision Making**  
  No need to wait for dashboards. Upload CSV, get insights, predictions, and PDF reports instantly.

- 📉 **Churn/Risk Prediction**  
  Use historical data to classify/predict churn, loan risk, payment defaults, etc.

- 🏠 **Real Estate or Finance Forecasting**  
  Upload datasets and get automated price predictions, loan risk evaluation, or investment scoring.

- 📊 **Client-Facing Reports**  
  Export PDF summaries with smart charts and AI-generated insights — perfect for presentations.

- ⏱️ **Rapid Prototyping**  
  Start with real data, build a model, and validate hypotheses — all without code changes.

---

## 🚀 Features

- 📁 Upload your own dataset (`.csv`)
- 📈 Visual Analysis: Histograms, Pie Charts, Scatter, Line, Bar, Bubble, etc.
- 🧠 AI Insight Summary using **Google Gemini API**
- 🤖 ML Training: Auto-detects classification or regression
- ✨ Predict on custom input
- 📄 Export to PDF report

---

## 🧰 Tech Stack

- `Python`, `Streamlit`
- `pandas`, `seaborn`, `matplotlib`
- `scikit-learn` (ML models)
- `fpdf` (PDF report)
- **Gemini API** (for AI-based data summaries)

---

## ⚙️ Run Locally

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/nlg-insight-app.git
cd nlg-insight-app
````

2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Add your Gemini API Key:

Create a `.env` file and add:

```
GEMINI_API_KEY=your_actual_key_here
```

Or directly set it in your `ai_summary.py` file if you're not using `.env`.

4. Start the app:

```bash
streamlit run app.py
```

---

## 📝 Project Structure

```
nlg-insight-app/
├── app.py
├── ai_summary.py        # Gemini API logic
├── ml_predictor.py      # Model training logic
├── pdf_generator.py     # Report generation
├── requirements.txt
└── README.md
```

---

## 🙌 Built By

**Anjali Pillai** – AI & Data Science Enthusiast
*“Turning raw data into smart decisions.”*

---

## 📄 License

MIT License. Free to use and adapt.

```

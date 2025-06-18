

# 📊 AutoInsight: AI-Powered Multi-Chart EDA Generator

**AutoInsight** is a smart and interactive Streamlit-based web app that enables users to:
- Upload any CSV dataset
- Generate 1 to 5 customizable visual charts (Bar, Histogram, Line, Boxplot)
- Automatically create professional AI-generated data summaries
- Download the entire report as a **PDF** (charts + insights)

📌 Whether you're a data analyst or a student, AutoInsight helps you turn raw data into polished insights instantly.

---

## 🚀 Live Demo

👉 [Click here to try AutoInsight: https://autoinsight-app-rufex8pp9axc8tmuyskx3l.streamlit.app)
---

## 🛠 Tech Stack

- **Frontend/UI**: Streamlit
- **Charts**: Seaborn, Matplotlib
- **AI Summary**: Python NLP (custom logic)
- **PDF Generation**: FPDF
- **Language**: Python

---

## 🔍 Features

- 📂 Upload any `.csv` file
- 📊 Generate **multiple visualizations** side-by-side
- 🎯 Choose your chart type and column for each chart
- 🧠 AI-generated human-readable insight summary
- 📄 One-click PDF report download (with all charts + summary)
- 🖼️ Clean, minimalist interface for better usability

---

## 📦 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/anjalipillai2603/autoinsight-app.git
cd autoinsight-app
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/9bd1501b-a952-4a82-8a2f-14635a701b29)

![image](https://github.com/user-attachments/assets/17ca3e56-a4e9-458e-9fc6-d66f9da88e93)

![image](https://github.com/user-attachments/assets/a070c3a2-8a93-4e09-af61-108e56399db9)

## 🧠 AI Insight Summary Generator

AutoInsight includes a custom-built summary generator using Python's NLP and pandas logic:

* Detects data shape and column types
* Calculates basic stats like mean, std, max
* Reports most frequent values for categorical fields
* Summary is returned in a **clean, readable format**

🔄 You can modify `ai_summary.py` to plug in GPT or other AI APIs later!

---

## 📁 Project Structure

```
autoinsight-app/
│
├── app.py               # Streamlit app
├── ai_summary.py        # Generates AI insight
├── pdf_generator.py     # Converts summary + charts into PDF
├── requirements.txt     # Dependency list
├── README.md            # You’re reading this
```

---

## 👩‍💻 Author

**Anjali Pillai**
| Aspiring Data Scientist | Analytics Enthusiast
🔗Let's Connect [LinkedIn](www.linkedin.com/in/anjali-pillai-367b2b259)

---

## 📜 License

This project is open-source and available under the [MIT License].

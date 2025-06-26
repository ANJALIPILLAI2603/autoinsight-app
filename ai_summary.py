import pandas as pd
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# 🔐 Replace with your actual Gemini API key
genai.configure(api_key="AIzaSyArhPvbuMHEEY4OKhP0hPjNYTOk3jJUmto")

def generate_structured_summary(df):
    rows, cols = df.shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    lines = [f"Final Summary: The dataset contains {rows} rows and {cols} columns."]

    if num_cols:
        lines.append(f"It includes {len(num_cols)} numeric columns such as {', '.join(num_cols[:3]) + ('...' if len(num_cols) > 3 else '')}.")
        lines.append("For numeric data, here are some highlights:")
        for col in num_cols:
            try:
                mean = round(df[col].mean(), 2)
                std = round(df[col].std(), 2)
                max_val = round(df[col].max(), 2)
                lines.append(f"- {col}: Mean = {mean}, Std = {std}, Max = {max_val}")
            except:
                continue

    if cat_cols:
        lines.append(f"It includes {len(cat_cols)} categorical columns like {', '.join(cat_cols[:3]) + ('...' if len(cat_cols) > 3 else '')}.")
        lines.append("For categorical data:")
        for col in cat_cols:
            try:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
                lines.append(f"- Most common value in '{col}' is '{mode_val}'")
            except:
                continue

    return "\n".join(lines)

def get_gemini_summary(text):
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(f"Summarize this dataset insight in a clean, readable format:\n\n{text}")
            return response.text.strip()
        except ResourceExhausted as e:
            print(f"[Retry {attempt+1}] Gemini quota exceeded. Waiting 30 seconds before retrying...")
            time.sleep(30)
    raise RuntimeError("Failed after multiple retries: Gemini API quota exceeded.")

def create_nlp_summary(df):
    structured = generate_structured_summary(df)
    ai_summary = get_gemini_summary(structured)
    return f"**AutoInsight Report**\n\n{ai_summary}"
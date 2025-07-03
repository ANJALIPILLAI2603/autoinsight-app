import os
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# 🔐 Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_structured_summary(df):
    rows, cols = df.shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    lines = [f"The dataset contains {rows} rows and {cols} columns."]

    if num_cols:
        lines.append(f"\nIt includes {len(num_cols)} numeric columns such as {', '.join(num_cols[:3]) + ('...' if len(num_cols) > 3 else '')}.\n")
        lines.append("**Numeric Column Insights:**")
        for col in num_cols:
            try:
                mean = round(df[col].mean(), 2)
                std = round(df[col].std(), 2)
                max_val = round(df[col].max(), 2)
                lines.append(f"{col} has an average of {mean:,.0f}, with a maximum of {max_val:,.0f}.")
                
            except:
                continue

    if cat_cols:
        lines.append(f"\nIt includes {len(cat_cols)} categorical columns like {', '.join(cat_cols[:3]) + ('...' if len(cat_cols) > 3 else '')}.\n")
        lines.append("**Categorical Column Insights:**")
        for col in cat_cols:
            try:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
                lines.append(f"- **{col}**: Most common value = '{mode_val}'")
            except:
                continue

    return "\n".join(lines)

def get_gemini_summary(text):
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(
                f"Summarize this dataset insight in a clean, readable format:\n\n{text}"
            )
            return response.text.strip()
        except ResourceExhausted:
            print(f"[Retry {attempt + 1}] Quota exceeded. Retrying in 30 seconds...")
            time.sleep(30)
        except Exception as e:
            print(f"[Retry {attempt + 1}] Unexpected error: {e}")
            time.sleep(15)
    raise RuntimeError("❌ Failed after 3 retries due to API quota or error.")

def create_nlp_summary(df):
    structured = generate_structured_summary(df)
    ai_summary = get_gemini_summary(structured)
    return f"**AutoInsight Report**\n\n{ai_summary}"

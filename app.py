import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ai_summary import create_nlp_summary
from pdf_generator import generate_pdf
import os

st.set_page_config(page_title="📊 AutoInsight: Multi-Chart EDA", layout="wide")
st.title("📊 AutoInsight: Multi-Chart Smart Report Generator")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📁 Data Preview")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.sidebar.header("📌 Select Multiple Visualization Options")
    num_charts = st.sidebar.slider("How many charts do you want?", 1, 5, 3)

    chart_inputs = []
    for i in range(num_charts):
        st.sidebar.markdown(f"### Chart {i+1}")
        chart_type = st.sidebar.selectbox(f"Chart Type {i+1}", ["Histogram", "Boxplot", "Bar", "Line"], key=f"type{i}")
        column = st.sidebar.selectbox(f"Column {i+1}", num_cols + cat_cols, key=f"col{i}")
        chart_inputs.append((chart_type, column))

    st.subheader("📈 Generated Visualizations")
    fig_paths = []

    for idx, (chart_type, selected_col) in enumerate(chart_inputs):
        st.markdown(f"#### Chart {idx+1}: {chart_type} of {selected_col}")
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller chart size

        if chart_type == "Histogram" and selected_col in num_cols:
            sns.histplot(df[selected_col], kde=True, ax=ax)
        elif chart_type == "Boxplot" and selected_col in num_cols:
            sns.boxplot(x=df[selected_col], ax=ax)
        elif chart_type == "Bar" and selected_col in cat_cols:
            df[selected_col].value_counts().head(10).plot(kind="bar", ax=ax)
        elif chart_type == "Line" and selected_col in num_cols:
            df[selected_col].plot(kind="line", ax=ax)

        ax.set_title(f"{chart_type} of {selected_col}")
        st.pyplot(fig)

        fig_path = f"plot_{idx}_{selected_col}_{chart_type}.png"
        fig.savefig(fig_path)
        fig_paths.append(fig_path)

    st.subheader("🧠 AI-Generated Insight Summary")
    summary = create_nlp_summary(df)
    st.markdown(summary, unsafe_allow_html=True)

    if st.button("📄 Download PDF Report"):
        pdf_path = generate_pdf(summary, fig_paths)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Click to Download", f, file_name="AutoInsight_Report.pdf")

    for path in fig_paths:
        os.remove(path)
else:
    st.info("📂 Please upload a CSV file to begin.")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ai_summary import create_nlp_summary
from pdf_generator import generate_pdf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import os
import numpy as np

st.set_page_config(page_title="📊 AutoInsight: Smart Report Generator", layout="wide")
st.title("📊 AutoInsight: Smart Report & ML Predictor")

# Tabs for EDA and ML
eda_tab, ml_tab = st.tabs(["📈 EDA + PDF Report", "🤖 ML Predictions"])

# ========================= EDA TAB =========================
with eda_tab:
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
            chart_type = st.sidebar.selectbox(f"Chart Type {i+1}", ["Histogram", "Boxplot", "Bar", "Line", "Scatter", "Pie", "Bubble"], key=f"type{i}")
            if chart_type == "Pie":
                col = st.sidebar.selectbox(f"Categorical Column {i+1}", cat_cols, key=f"piecol{i}")
                chart_inputs.append((chart_type, col, None))
            elif chart_type == "Bubble":
                x_col = st.sidebar.selectbox(f"X-Axis {i+1}", num_cols, key=f"x{i}")
                y_col = st.sidebar.selectbox(f"Y-Axis {i+1}", num_cols, key=f"y{i}")
                size_col = st.sidebar.selectbox(f"Bubble Size {i+1}", num_cols, key=f"size{i}")
                chart_inputs.append((chart_type, x_col, y_col, size_col))
            elif chart_type == "Scatter":
                x_col = st.sidebar.selectbox(f"X-Axis {i+1}", num_cols, key=f"x{i}")
                y_col = st.sidebar.selectbox(f"Y-Axis {i+1}", num_cols, key=f"y{i}")
                chart_inputs.append((chart_type, x_col, y_col))
            else:
                col = st.sidebar.selectbox(f"Column {i+1}", num_cols + cat_cols, key=f"col{i}")
                chart_inputs.append((chart_type, col))

        st.subheader("📈 Generated Visualizations")
        fig_paths = []

        for idx, chart_input in enumerate(chart_inputs):
            chart_type = chart_input[0]
            st.markdown(f"#### Chart {idx+1}: {chart_type}")
            fig, ax = plt.subplots()

            if chart_type == "Histogram":
                sns.histplot(df[chart_input[1]], kde=True, ax=ax, color="skyblue")
            elif chart_type == "Boxplot":
                sns.boxplot(x=df[chart_input[1]], ax=ax, palette="Set2")
            elif chart_type == "Bar":
                df[chart_input[1]].value_counts().head(10).plot(kind="bar", ax=ax, color="orange")
            elif chart_type == "Line":
                df[chart_input[1]].plot(kind="line", ax=ax, color="green")
            elif chart_type == "Pie":
                df[chart_input[1]].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax, cmap='Set3')
            elif chart_type == "Scatter":
                sns.scatterplot(data=df, x=chart_input[1], y=chart_input[2], ax=ax, palette="husl")
            elif chart_type == "Bubble":
                ax.scatter(df[chart_input[1]], df[chart_input[2]], s=df[chart_input[3]], alpha=0.5, c=np.random.rand(len(df)), cmap='viridis')

            ax.set_title(f"{chart_type} Chart")
            st.pyplot(fig)

            path = f"chart_{idx}.png"
            fig.savefig(path)
            fig_paths.append(path)

        st.subheader("🧠 AI-Generated Insight Summary")
        summary = create_nlp_summary(df)
        st.markdown(summary, unsafe_allow_html=True)

        if st.button("📄 Download PDF Report"):
            pdf_path = generate_pdf(summary, fig_paths)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Report", f, file_name="AutoInsight_Report.pdf")

        for path in fig_paths:
            os.remove(path)

    else:
        st.info("📂 Please upload a CSV file to begin.")

# ========================= ML TAB =========================
with ml_tab:
    st.subheader("🤖 Auto ML Prediction")
    ml_file = st.file_uploader("📂 Upload a CSV file for ML training", type=["csv"], key="ml")

    if ml_file:
        df = pd.read_csv(ml_file)
        st.success("✅ File loaded!")
        st.dataframe(df.head())

        # Drop ID-like or name-like columns from selection
        exclude_cols = [col for col in df.columns if any(x in col.lower() for x in ["id", "name", "number", "row"])]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        target = st.selectbox("🎯 Select Target Column", feature_cols)
        input_features = [col for col in feature_cols if col != target]

        if target:
            X = df[input_features]
            y = df[target]

            # Save original categorical values for UI
            cat_inputs = {col: df[col].unique().tolist() for col in X.select_dtypes(include="object").columns}

            X = pd.get_dummies(X, drop_first=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier() if y.nunique() <= 10 and y.dtype == 'int' else RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.session_state.model = model
            st.session_state.features = X.columns.tolist()
            st.session_state.cat_inputs = cat_inputs
            st.session_state.raw_inputs = input_features

            st.subheader("📊 Model Results")
            if isinstance(model, RandomForestClassifier):
                st.write("✅ Accuracy:", accuracy_score(y_test, preds))
                st.text("Classification Report:\n" + classification_report(y_test, preds))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, preds))
            else:
                st.write("✅ MAE:", mean_absolute_error(y_test, preds))
                st.write("✅ RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

            result_df = pd.DataFrame({
                f"{target}_actual": y_test.values,
                f"{target}_predicted": preds
            })

            st.subheader("📤 Download Predictions")
            st.dataframe(result_df.head())
            result_df.to_csv("predictions.csv", index=False)
            with open("predictions.csv", "rb") as f:
                st.download_button("📥 Download CSV", f, file_name="predictions.csv")

    # ================= Predict with Unseen Data =================
    if "model" in st.session_state and "features" in st.session_state:
        st.subheader("🧑‍💻 Predict with New Data")

        user_inputs = {}
        for col in st.session_state.raw_inputs:
            if col in st.session_state.cat_inputs:
                user_inputs[col] = st.selectbox(f"{col}", st.session_state.cat_inputs[col])
            else:
                user_inputs[col] = st.text_input(f"{col}", value="")

        if st.button("🔮 Predict"):
            try:
                input_df = pd.DataFrame([user_inputs])
                input_encoded = pd.get_dummies(input_df)

                for col in st.session_state.features:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0

                input_encoded = input_encoded[st.session_state.features]
                prediction = st.session_state.model.predict(input_encoded)[0]
                st.success(f"📍 Predicted value: **{prediction}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

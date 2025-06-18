def create_nlp_summary(df):
    rows, cols = df.shape
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    summary = f"✅ Final Summary: The dataset contains {rows} rows and {cols} columns.\n"

    if num_cols:
        summary += f"\nIt includes {len(num_cols)} numeric columns such as {', '.join(num_cols[:3])}."
        summary += "\n\n📌 For numeric data, here are some highlights:\n"
        for col in num_cols[:3]:
            mean = round(df[col].mean(), 2)
            std = round(df[col].std(), 2)
            max_val = round(df[col].max(), 2)
            summary += f"- {col}: Mean = {mean}, Std = {std}, Max = {max_val}\n"

    if cat_cols:
        summary += f"\nIt includes {len(cat_cols)} categorical columns like {', '.join(cat_cols[:3])}."
        summary += "\n\n📌 For categorical data:\n"
        for col in cat_cols[:3]:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
            summary += f"- Most common value in '{col}' is '{mode_val}'\n"

    return summary

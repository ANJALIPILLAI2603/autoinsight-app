def generate_structured_summary(df):
    rows, cols = df.shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    lines = [
        f"Final Summary: The dataset contains {rows} rows and {cols} columns.",
    ]

    if num_cols:
        lines.append(f"It includes {len(num_cols)} numeric columns such as {', '.join(num_cols[:3]) + ('...' if len(num_cols) > 3 else '')}.")
        lines.append("For numeric data, here are some highlights:")
        for col in num_cols[:3]:
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
        for col in cat_cols[:3]:
            try:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
                lines.append(f"- Most common value in '{col}' is '{mode_val}'")
            except:
                continue

    return "\n".join(lines)

def create_nlp_summary(df):
    structured = generate_structured_summary(df)
    return f"**AutoInsight Report**\n\n**AI-Powered Data Summary**\n\n{structured}"

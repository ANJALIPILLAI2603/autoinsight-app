from fpdf import FPDF
import textwrap
import os
import re

def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "AutoInsight Report", ln=1, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_summary(self, summary):
        self.set_font("Arial", "", 12)
        clean_text = remove_emojis(summary)
        paragraphs = clean_text.split("\n")

        for para in paragraphs:
            wrapped = textwrap.wrap(para.strip(), 90)
            for line in wrapped:
                self.cell(0, 8, line, ln=1)
            self.ln(2)  # Space between paragraphs

    def add_image(self, image_path):
        if os.path.exists(image_path):
            self.ln(10)
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Visualization", ln=1)
            self.image(image_path, w=150)  # Reduced size
        else:
            self.cell(0, 10, "Chart image not found.", ln=1)

def generate_pdf(summary, image_paths):
    pdf = PDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI-Powered Data Summary", ln=1)
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.add_summary(summary)

    for img_path in image_paths:
        pdf.ln(10)
        pdf.add_image(img_path)

    output_path = "AutoInsight_Report.pdf"
    pdf.output(output_path)
    return output_path

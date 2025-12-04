import pypdf
import sys

def extract_text_from_pdf(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = r"d:\75128\Desktop\2025秋_课程_大三上\具身人工智能\EAI-CourseProject-LeRobot\Project Announcement.pdf"
    print(extract_text_from_pdf(pdf_path))

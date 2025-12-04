
def dump_pdf_text(pdf_path, output_path):
    try:
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        # Filter for printable characters
        text = ""
        for byte in content:
            if 32 <= byte <= 126 or byte == 10 or byte == 13:
                text += chr(byte)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
    except Exception as e:
        print(e)

if __name__ == "__main__":
    pdf_path = r"d:\75128\Desktop\2025秋_课程_大三上\具身人工智能\EAI-CourseProject-LeRobot\Project Announcement.pdf"
    output_path = r"d:\75128\Desktop\2025秋_课程_大三上\具身人工智能\EAI-CourseProject-LeRobot\pdf_dump.txt"
    dump_pdf_text(pdf_path, output_path)

import re
import zlib

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        content = f.read()

    # Find all streams
    # PDF streams are marked by 'stream\r\n' or 'stream\n' and 'endstream'
    # We'll use a regex to find them.
    
    # This regex is a bit simplistic but might work for this file.
    # We look for the keyword stream followed by newline, then content, then endstream.
    stream_pattern = re.compile(rb'stream[\r\n]+(.*?)[\r\n]+endstream', re.DOTALL)
    
    streams = stream_pattern.findall(content)
    
    extracted_text = ""
    
    print(f"Found {len(streams)} streams.")
    
    for i, stream_data in enumerate(streams):
        try:
            # Try to decompress
            decompressed_data = zlib.decompress(stream_data)
            # Try to decode as utf-8 or latin-1
            try:
                text = decompressed_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = decompressed_data.decode('latin-1')
                except:
                    text = str(decompressed_data)
            
            # Filter for readable text (simple heuristic)
            # We are looking for text inside BT (Begin Text) and ET (End Text) blocks usually,
            # or just raw text in parenthesis like (Hello World) or Tj operators.
            
            # Let's just dump everything that looks like text for now.
            # PDF text is often in parentheses (...) or brackets <...>
            # But often it's just mixed in commands.
            
            # A simple way is to look for strings in parentheses
            text_fragments = re.findall(r'\((.*?)\)', text)
            if text_fragments:
                extracted_text += f"\n--- Stream {i} ---\n"
                for fragment in text_fragments:
                    # Unescape PDF strings
                    fragment = fragment.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\(', '(').replace('\\)', ')')
                    extracted_text += fragment + "\n"
            
            # Also look for TJ arrays which are like [(Text) 20 (Text)] TJ
            # This is harder to parse with regex, but let's try to capture simple text.
            
        except zlib.error:
            # Not compressed or different compression
            pass
        except Exception as e:
            # print(f"Error processing stream {i}: {e}")
            pass

    return extracted_text

if __name__ == "__main__":
    pdf_path = "Track 1 Details.pdf"
    text = extract_text_from_pdf(pdf_path)
    
    with open("track1_details_extracted.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    print("Extraction complete. Check pdf_content_extracted.txt")

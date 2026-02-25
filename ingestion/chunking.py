import os
import json
import re

CHUNK_SIZE = 150
OVERLAP = 30
RAW_DIR= "data/raw"
PROCESSED_DIR = "data/processed" 

import os
import json
import re

CHUNK_SIZE = 1000
OVERLAP = 200


def chunk_text(text, company, year, source):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "company": company,
                "year": year,
                "chunk_id": len(chunks),
                "source": source
            }
        })

        start += CHUNK_SIZE - OVERLAP

    return chunks


def chunk_file(file_path: str, company: str, year: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    
    sections = re.split(r"(ITEM\s+\d+[A-Z]?\.)", text, flags=re.IGNORECASE)

    all_chunks = []
    source = os.path.basename(file_path)

    
    for i in range(1, len(sections), 2):
        section_title = sections[i]
        section_text = sections[i + 1] if i + 1 < len(sections) else ""

        combined_text = section_title + " " + section_text

        section_chunks = chunk_text(combined_text, company, year, source)

        # Add section metadata
        for chunk in section_chunks:
            chunk["metadata"]["section"] = section_title.strip()

        all_chunks.extend(section_chunks)

    return all_chunks

def process_all():
    os.makedirs(PROCESSED_DIR,exist_ok=True)
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith('.txt'):
            continue
        parts = filename.replace('.txt',"").split("_")
        company,year = parts[0],parts[2][:4]
        filepath = os.path.join(RAW_DIR,filename)
        chunks = chunk_file(filepath,company,year)
        out_path = os.path.join(PROCESSED_DIR,filename.replace(".txt",".json"))
        with open(out_path,"w") as f:
            json.dump(chunks,f,indent=2)
        print(f"Chunked{filename}->{len(chunks)} chunks")


if __name__ == "__main__":
    process_all()
    
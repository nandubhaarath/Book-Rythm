import re
import sys

import fitz


def extract_paragraphs(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = " ".join(page.get_text() for page in doc)
    doc.close()

    text = text.replace("\n", " ")
    text = re.sub(r" {2,}", " ", text).strip()

    sentences = re.split(r"(?<=[.!?]) ", text)

    chunks = []
    current = ""
    for sentence in sentences:
        current = f"{current} {sentence}".strip() if current else sentence
        if len(current) > 400:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)

    result = []
    for chunk in chunks:
        if len(chunk) < 50:
            if result:
                result[-1] = result[-1] + " " + chunk
            continue
        result.append(chunk)

    return result


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        pdf_bytes = f.read()

    chunks = extract_paragraphs(pdf_bytes)
    total = len(chunks)
    average_length = sum(len(c) for c in chunks) / total if total else 0

    print(f"Total chunks: {total}")
    print(f"Average chunk length: {average_length:.1f}")

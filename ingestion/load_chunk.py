from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

DATA_FOLDER = "./data"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 90  # 15% of chunk_size

def extract_text_from_pdf(filepath):
    """Extract text page-by-page, skipping the first 2 
    pages (cover/promotional content), tagging each chunk 
    with its source page number."""
    reader = PdfReader(filepath)
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        if page_num < 2:  # skip pages 1-2 (0-indexed: 0,1)
            continue
        text = page.extract_text()
        if text.strip():
            pages_text.append({
                "page_number": page_num + 1,
                "text": text
            })
    return pages_text

def is_low_value_chunk(text):
    """Filter out promotional/non-content text."""
    noise_keywords = ["Batch Starts", "info@", "pwonlyias.com", 
                       "Test Series", "DPP", "Prelims + Mains"]
    if any(kw in text for kw in noise_keywords):
        return True
    if len(text.strip()) < 50:  # too short to be real content
        return True
    return False    

def chunk_pages(pages_text, source_filename):
    """Split each page's text into chunks, attaching 
    metadata so we know exactly where each chunk came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = []
    for page in pages_text:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            if is_low_value_chunk(chunk):
                continue
            all_chunks.append({
                "text": chunk,
                "source": source_filename,
                "page": page["page_number"],
                "chunk_index": i
            })
    return all_chunks

def main():
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s) in {DATA_FOLDER}")

    all_chunks = []
    for pdf_file in pdf_files:
        filepath = os.path.join(DATA_FOLDER, pdf_file)
        print(f"Processing: {pdf_file}")
        pages_text = extract_text_from_pdf(filepath)
        chunks = chunk_pages(pages_text, pdf_file)
        all_chunks.extend(chunks)
        print(f"  -> {len(pages_text)} pages, {len(chunks)} chunks")

    print(f"\nTotal chunks across all PDFs: {len(all_chunks)}")
    print("\n--- Sample chunk ---")
    print(all_chunks[0])

if __name__ == "__main__":
    main()
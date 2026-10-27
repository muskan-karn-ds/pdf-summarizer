import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load the API key from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 12000  # characters per chunk


def chunk_text(text, size=CHUNK_SIZE):
    """Split text into chunks of roughly `size` characters, at word boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # try not to cut a word in half
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space != -1:
                end = last_space
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def summarize_chunk(chunk, index, total):
    """Summarize one chunk."""
    prompt = (
        f"This is part {index} of {total} of a document. "
        f"Summarize it clearly and concisely:\n\n{chunk}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You summarize text accurately and concisely."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def summarize_all(chunks):
    """Summarize all chunks, then combine them."""
    if len(chunks) == 1:
        return summarize_chunk(chunks[0], 1, 1)

    partial_summaries = []
    progress = st.progress(0, text="Summarizing sections...")
    for i, chunk in enumerate(chunks, start=1):
        partial_summaries.append(summarize_chunk(chunk, i, len(chunks)))
        progress.progress(i / len(chunks), text=f"Summarized {i}/{len(chunks)} sections")

    progress.empty()

    # Final pass: combine the partial summaries
    combined = "\n\n".join(partial_summaries)
    final_prompt = (
        "Below are summaries of consecutive parts of a single document. "
        "Combine them into one clear, coherent overall summary:\n\n" + combined
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You combine partial summaries into one clear overall summary."},
            {"role": "user", "content": final_prompt},
        ],
    )
    return response.choices[0].message.content


st.title("PDF Summarizer Bot")
st.write("Upload a PDF and I'll summarize it.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    st.success(f"PDF read successfully! ({len(reader.pages)} pages, {len(text)} characters)")

    with st.expander("Preview extracted text (first 1000 characters)"):
        st.write(text[:1000])

    if st.button("Summarize"):
        if not text.strip():
            st.warning("No text could be extracted from this PDF. It may be a scanned document.")
        else:
            chunks = chunk_text(text)
            st.info(f"Document split into {len(chunks)} section(s).")
            try:
                summary = summarize_all(chunks)
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error: {e}")
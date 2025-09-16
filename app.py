import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# -----------------------------
# Load resume text
# -----------------------------
with open("my_data.md", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(resume_text)

# -----------------------------
# Embeddings & Vector DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Local QA model (lightweight)
# -----------------------------
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def ask_bot(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a helpful assistant. Answer the following question
    ONLY using Dhruv Kannojia's resume information below:

    {context}

    Question: {query}
    Answer:
    """
    response = qa_pipeline(prompt, max_new_tokens=200)
    return response[0]["generated_text"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💼 Dhruv Kannojia - Resume Chatbot")
st.write("Ask me anything about Dhruv!")

query = st.text_input("Your question:")
if query:
    answer = ask_bot(query)
    st.write(answer)

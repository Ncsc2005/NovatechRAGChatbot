import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
DATA_FOLDER=os.path.join(os.path.dirname(__file__),'_data')
st.set_page_config(
    page_title="NovaTech Assistant",
    page_icon="🤖",
    layout="centered"
    )
st.title("NovaTech Internal Assistant")
st.caption("Ask me anything about Novatech company Policies,Products or Procedures")
st.divider()
@st.cache_resource
def load_rag():
    if not os.path.exists(DATA_FOLDER):
        st.error(f"_data/folder not found at:{DATA_FOLDER}")
        st.stop()
    chunks  = []  
    sources = []   
    for filename in sorted(os.listdir(DATA_FOLDER)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DATA_FOLDER,filename),"r", encoding="utf-8") as f:
            text=f.read()
        for para in text.strip().split("\n\n"):
            para=para.strip()
            if len(para) < 50:
                continue
            if para.startswith("==="):
                continue
            chunks.append(para)
            sources.append(filename)
            
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    return chunks, sources, vectorizer, tfidf_matrix, client
with st.spinner("Loading and indexing company documents...."):
    chunks, sources, vectorizer, tfidf_matrix, groq_client = load_rag()
st.success(f"Ready — {len(chunks)} document chunks indexed.", icon="✅")
st.divider()
if "messages"not in st.session_state:
    st.session_state.messages=[]

def ask_rag(question: str, top_k: int = 3) -> dict:
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_chunks  = [chunks[i]           for i in top_indices]
    retrieved_sources = [sources[i]          for i in top_indices]
    retrieved_scores  = [float(similarities[i]) for i in top_indices]
    context = "\n\n---\n\n".join(retrieved_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful NovaTech company assistant. "
                "Answer questions using ONLY the provided context. "
                "If the context does not contain enough information, "
                "say 'I don't have enough information to answer this.' "
                "Be concise and direct."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "answer":  answer,
        "sources": list(dict.fromkeys(retrieved_sources)),        
        "chunks":  list(zip(retrieved_chunks, retrieved_sources, retrieved_scores))
    }        
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("sources"):
                st.caption(f"Sources: {', '.join(msg['sources'])}")

            if msg.get("chunks"):
                with st.expander("View retrieved document chunks"):
                    for i, (chunk_text, source_file, score) in enumerate(msg["chunks"], 1):
                        st.markdown(
                            f"**Chunk {i} — `{source_file}` (Score: {score:.2f})**"
                        )
                        st.info(chunk_text)
question = st.chat_input("Ask a question about NovaTech policies or products...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = ask_rag(question)
        st.markdown(result["answer"])
        st.caption(f"Sources: {', '.join(result['sources'])}")
        with st.expander("View retrieved document chunks"):
            for i, (chunk_text, source_file, score) in enumerate(result["chunks"], 1):
                st.markdown(f"**Chunk {i} — `{source_file}` (Score: {score:.2f})**")
                st.info(chunk_text)
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks":  result["chunks"]
    })
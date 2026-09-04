import os
import streamlit as st
import requests
import json

# backend URL (set to http://backend:8000 in docker, localhost for local run)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Week 15 AI Assistant", layout="wide")

st.title("Week 15 AI Assistant")
st.caption("Frontend interface for FastAPI backend with RAG, tool calling, and fallback.")

# sidebar settings
st.sidebar.header("Configuration")

# system prompt dropdown
system_prompts = {
    "General Assistant": "You are a helpful and polite AI assistant.",
    "Code Reviewer": "You are a concise code evaluator. Analyze code directly.",
    "Fellowship Mentor": "You are an experienced mentor guiding AI Fellowship students.",
    "Brief & Direct": "Answer concisely in two sentences or less."
}
selected_prompt_name = st.sidebar.selectbox(
    "System Prompt",
    list(system_prompts.keys())
)
chosen_system_prompt = system_prompts[selected_prompt_name]

# model choice
models = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "meta-llama/Meta-Llama-3-8B-Instruct (vLLM)",
    "mistralai/Mistral-7B-Instruct-v0.2 (vLLM)"
]
selected_model_option = st.sidebar.selectbox("Model", models)
model_name = selected_model_option.split(" ")[0]

# generation parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
top_p = st.sidebar.slider("Top-P", min_value=0.1, max_value=1.0, value=1.0, step=0.05)

# toggles
use_tool = st.sidebar.checkbox("Enable Weather Tool (Function Calling)", value=False)
use_structured = st.sidebar.checkbox("Use Structured JSON Output (Pydantic)", value=False)

# backend health check indicator
st.sidebar.markdown("---")
try:
    health_res = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend: Online (Port 8000)")
    else:
        st.sidebar.warning(f"Backend status: {health_res.status_code}")
except Exception:
    st.sidebar.error("Backend: Offline (Start main.py)")


# tabs for chat, rag, and ingestion
tab1, tab2, tab3 = st.tabs(["Chat & Tools", "RAG Search", "Ingest Documents"])

with tab1:
    st.subheader("Direct Chat")
    user_prompt = st.text_area("User Message", placeholder="Ask something, e.g. 'What is the weather in Kathmandu?'")

    if st.button("Send", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter a message.")
        else:
            payload = {
                "prompt": user_prompt,
                "system_prompt": chosen_system_prompt,
                "model": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "use_tool": use_tool,
                "use_structured_output": use_structured
            }

            try:
                res = requests.post(f"{BACKEND_URL}/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.info(f"Model used: {data.get('model')}")

                    if "tool_called" in data:
                        st.markdown(f"**Tool Called:** `{data['tool_called']}`")
                        st.write("Tool Arguments:", data.get("tool_args"))
                        st.write("Tool Result:", data.get("tool_result"))

                    if data.get("structured_data"):
                        st.markdown("**Structured Output (Pydantic):**")
                        st.json(data["structured_data"])

                    st.markdown("**Response:**")
                    st.write(data.get("content"))
                elif res.status_code == 429:
                    st.error("Rate limit reached. Please wait a moment before sending another request.")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab2:
    st.subheader("RAG Document Q&A")
    rag_query = st.text_input("Question", placeholder="e.g. What does Week 15 focus on?")

    if st.button("Search & Answer"):
        if not rag_query.strip():
            st.warning("Please enter a question.")
        else:
            payload = {
                "question": rag_query,
                "temperature": temperature,
                "top_p": top_p
            }
            try:
                res = requests.post(f"{BACKEND_URL}/rag-chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.info(f"Model used: {data.get('model')}")
                    st.markdown("**Answer:**")
                    st.write(data.get("answer"))

                    st.markdown("**Retrieved Context from ChromaDB:**")
                    retrieved = data.get("retrieved_context", [])
                    if retrieved:
                        for i, doc in enumerate(retrieved):
                            src = doc.get("metadata", {}).get("source", "Unknown")
                            with st.expander(f"Chunk {i+1} - Source: {src}"):
                                st.write(doc.get("text"))
                    else:
                        st.write("No matching documents found.")
                elif res.status_code == 429:
                    st.error("Rate limit reached. Please wait a moment.")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab3:
    st.subheader("Document Ingestion")
    st.write("Load text documents from a directory and index them in ChromaDB.")

    docs_dir = st.text_input("Folder path", value="./docs")

    if st.button("Run Ingestion"):
        try:
            res = requests.post(f"{BACKEND_URL}/ingest", params={"folder_path": docs_dir})
            if res.status_code == 200:
                count = res.json().get("chunks_ingested", 0)
                st.success(f"Ingestion complete: {count} chunks added to ChromaDB.")
            else:
                st.error(f"Ingestion failed: {res.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

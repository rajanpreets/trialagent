import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import requests
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# --- 1. CONFIGURATION ---

# Configure the Gemini API key from Streamlit secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
    
# --- UPDATE THESE URLS ---
# Replace with the direct RAW links to your files on GitHub
FAISS_INDEX_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/optimized_trial_index.faiss"
METADATA_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/trial_metadata.csv"

# Model configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATIVE_MODEL_NAME = 'gemini-1.5-pro-latest' # Using the latest powerful model available

# --- 2. DATA AND MODEL LOADING (CACHED) ---

@st.cache_resource
def load_data_and_models(index_url, metadata_url):
    """
    Downloads data from GitHub and loads models into memory.
    This function is cached to run only once.
    """
    local_data_path = "data"
    if not os.path.exists(local_data_path):
        os.makedirs(local_data_path)

    local_index_path = os.path.join(local_data_path, "optimized_trial_index.faiss")
    local_metadata_path = os.path.join(local_data_path, "trial_metadata.csv")

    # Download files from GitHub if they don't exist locally
    with st.spinner("Downloading data files from GitHub (first time only)..."):
        if not os.path.exists(local_index_path):
            r = requests.get(index_url)
            with open(local_index_path, 'wb') as f:
                f.write(r.content)
        
        if not os.path.exists(local_metadata_path):
            r = requests.get(metadata_url)
            with open(local_metadata_path, 'wb') as f:
                f.write(r.content)
    
    # Load models and data into memory
    with st.spinner("Loading models and search index..."):
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        search_index = faiss.read_index(local_index_path)
        metadata_df = pd.read_csv(local_metadata_path)
        
    return embedding_model, search_index, metadata_df

# --- 3. THE RETRIEVER TOOL ---

def search_clinical_trials(query: str, k: int = 5):
    """The tool our agent will use to search for relevant trials."""
    embedding_model, search_index, metadata_df = st.session_state.models_and_data
    
    query_vector = embedding_model.encode([query]).astype('float32')
    distances, indices = search_index.search(query_vector, k)
    
    if len(indices[0]) == 0:
        return "No relevant trials found."

    results_df = metadata_df.iloc[indices[0]]
    
    # Format the results into a string for the LLM
    formatted_results = []
    for i, row in results_df.iterrows():
        result = (
            f"Trial ID: {row.get('protocolSection.identificationModule.nctId', 'N/A')}\n"
            f"Title: {row.get('protocolSection.identificationModule.officialTitle', 'N/A')}\n"
            f"Status: {row.get('protocolSection.statusModule.overallStatus', 'N/A')}\n"
            f"Summary: {row.get('protocolSection.descriptionModule.briefSummary', 'N/A')}\n"
        )
        formatted_results.append(result)
        
    return "\n---\n".join(formatted_results)

# --- 4. LANGGRAPH AGENT SETUP ---

class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str

# Define the nodes for the graph
def retrieve_node(state: AgentState):
    """Node that calls our search tool."""
    print("---NODE: RETRIEVE---")
    question = state["question"]
    documents = search_clinical_trials(question)
    return {"documents": documents}

def generate_node(state: AgentState):
    """Node that generates the final answer using Gemini."""
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = f"""You are a helpful assistant specializing in clinical trial data.
    Answer the user's question based *only* on the provided search results.
    If the results do not contain the answer, state that you cannot answer based on the information provided.

    USER QUESTION: {question}

    SEARCH RESULTS:
    {documents}
    """
    
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return {"answer": response.text}

def fallback_node(state: AgentState):
    """Node to handle cases where no documents are found."""
    print("---NODE: FALLBACK---")
    return {"answer": "I'm sorry, but I couldn't find any relevant clinical trials for your query. Please try rephrasing your question."}

# Define the conditional edge
def should_generate(state: AgentState):
    """Determines whether to generate an answer or use the fallback."""
    print("---EDGE: CONDITIONAL---")
    if state["documents"] and "No relevant trials found." not in state["documents"]:
        return "generate"
    else:
        return "fallback"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    should_generate,
    {"generate": "generate", "fallback": "fallback"},
)
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

app_graph = workflow.compile()

# --- 5. STREAMLIT UI ---

st.title("Clinical Trials Search Agent 🩺")
st.info("Powered by LangGraph and Gemini 1.5 Pro")

# Load models and data only once and store in session state
if 'models_and_data' not in st.session_state:
    st.session_state.models_and_data = load_data_and_models(FAISS_INDEX_URL, METADATA_URL)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about clinical trials..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run the LangGraph agent
            inputs = {"question": prompt}
            final_state = app_graph.invoke(inputs)
            response = final_state.get("answer", "An error occurred.")
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from groq import Groq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# --- 1. CONFIGURATION ---
try:
    # Set up the Groq API key from Streamlit secrets
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Groq API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
    
# Model configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'llama3-8b-8192' # Groq model for fast inference

# --- 2. DATA AND MODEL LOADING ---
@st.cache_resource
def load_models_and_data(faiss_file_obj, metadata_file_obj):
    """Loads models and data from user-uploaded file objects."""
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    faiss_path = os.path.join(temp_dir, faiss_file_obj.name)
    metadata_path = os.path.join(temp_dir, metadata_file_obj.name)
    with open(faiss_path, "wb") as f: f.write(faiss_file_obj.getbuffer())
    with open(metadata_path, "wb") as f: f.write(metadata_file_obj.getbuffer())
    with st.spinner("Loading models and search index..."):
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        search_index = faiss.read_index(faiss_path)
        metadata_df = pd.read_csv(metadata_path)
    return embedding_model, search_index, metadata_df

# --- 3. THE UPGRADED RETRIEVER TOOL ---
def search_clinical_trials(query: str, filters: dict, k_final: int):
    """Performs advanced search with semantic and structured filtering."""
    embedding_model, search_index, metadata_df = st.session_state.models_and_data
    working_df = metadata_df

    # Apply structured filters first
    if filters:
        for column, value in filters.items():
            actual_col = next((c for c in metadata_df.columns if column in c.lower()), None)
            if actual_col and value:
                working_df = working_df[working_df[actual_col].str.contains(value, case=False, na=False)]

    if working_df.empty:
        return pd.DataFrame()

    # If there's a semantic query, perform vector search
    if query:
        filtered_indices = working_df.index.to_numpy()
        query_vector = embedding_model.encode([query]).astype('float32')
        # Retrieve a large number of initial candidates for good overlap
        distances, initial_indices = search_index.search(query_vector, k=300)
        final_indices = [idx for idx in initial_indices[0] if idx in filtered_indices]
        results_df = metadata_df.iloc[final_indices[:k_final]]
    else:
        results_df = working_df.head(k_final)
        
    return results_df

# --- 4. LANGGRAPH AGENT SETUP ---
class AgentState(TypedDict):
    question: str
    k: int
    plan: dict
    retrieved_df: Optional[pd.DataFrame]
    batch_summaries: List[str]
    final_answer: str
    current_batch: int

# --- Agent Nodes ---
def planner_node(state: AgentState):
    """Decomposes the user query into a semantic component and structured filters using Groq."""
    prompt = f"""You are a query analysis expert. Your job is to break down a user's question about clinical trials into a semantic search query and structured filters.

    Identify the following filters if they exist:
    - sponsor (e.g., pfizer, moderna)
    - status (e.g., recruiting, completed)
    - phase (e.g., phase 1, phase 2)

    The remaining part of the query is the semantic search query.

    User Question: "{state['question']}"

    Provide your answer ONLY as a valid JSON object with two keys: "query" and "filters". The "filters" value should be another JSON object.

    Example:
    User Question: "recruiting pfizer trials for lung cancer"
    Your Response:
    {{
        "query": "lung cancer",
        "filters": {{
            "sponsor": "pfizer",
            "status": "recruiting"
        }}
    }}
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
    )
    plan_text = chat_completion.choices[0].message.content
    plan = json.loads(plan_text)
    return {"plan": plan}

def retrieve_node(state: AgentState):
    """Retrieves the full set of documents based on the plan."""
    results_df = search_clinical_trials(
        query=state["plan"].get("query", ""),
        filters=state["plan"].get("filters", {}),
        k_final=state["k"]
    )
    return {"retrieved_df": results_df}

def batch_summarize_node(state: AgentState):
    """Summarizes one batch of results."""
    df = state["retrieved_df"]
    start_index = state["current_batch"] * 5
    end_index = start_index + 5
    batch_df = df.iloc[start_index:end_index]
    
    if batch_df.empty:
        return {"batch_summaries": state.get("batch_summaries", [])}

    context = "\n---\n".join([f"Trial {i+1}:\n{row.to_string()}" for i, row in batch_df.iterrows()])
    prompt = f"Summarize the key findings from the following batch of clinical trials:\n\n{context}"
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL_NAME,
        temperature=0.2,
    )
    summary = chat_completion.choices[0].message.content
    
    new_summaries = state.get("batch_summaries", []) + [summary]
    return {"batch_summaries": new_summaries, "current_batch": state["current_batch"] + 1}

def final_summary_node(state: AgentState):
    """Creates a final summary from all batch summaries."""
    if not state.get("batch_summaries"):
        return {"final_answer": "No relevant trials were found based on your query."}
        
    context = "\n\n".join(state["batch_summaries"])
    prompt = f"""You are a research assistant. The following are summaries from batches of clinical trials. Synthesize them into a single, cohesive final answer for the user's original question. Provide a high-level overview first, then present the key details.

    User Question: "{state['question']}"

    Batch Summaries:
    {context}
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL_NAME,
        temperature=0.3,
    )
    final_summary = chat_completion.choices[0].message.content
    return {"final_answer": final_summary}

# --- Conditional Edges ---
def should_process(state: AgentState):
    if state["retrieved_df"] is None or state["retrieved_df"].empty:
        return "end"
    elif len(state["retrieved_df"]) > 10: # Threshold for batch processing
        return "start_batch_summary"
    else:
        # For small results, we can treat the whole set as one batch
        return "generate_direct_summary"

def more_batches(state: AgentState):
    if state["current_batch"] * 5 >= len(state["retrieved_df"]):
        return "to_final_summary"
    else:
        return "continue_batching"

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("batch_summarizer", batch_summarize_node)
workflow.add_node("final_summarizer", final_summary_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_conditional_edges(
    "retriever",
    should_process,
    {"end": END, "start_batch_summary": "batch_summarizer", "generate_direct_summary": "batch_summarizer"}
)
workflow.add_conditional_edges(
    "batch_summarizer",
    more_batches,
    {"to_final_summary": "final_summarizer", "continue_batching": "batch_summarizer"}
)
workflow.add_edge("final_summarizer", END)
app_graph = workflow.compile()

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("Advanced Clinical Trials Research Assistant 🔬")
st.info("Powered by LangGraph and Groq (Llama3-8B)")

# Sidebar
st.sidebar.header("1. Upload Your Data")
faiss_file = st.sidebar.file_uploader("Upload FAISS Index", type=["faiss"])
csv_file = st.sidebar.file_uploader("Upload CSV Metadata", type=["csv"])
st.sidebar.header("2. Search Settings")
k_results = st.sidebar.slider("Max results to display:", 1, 50, 10)

# Main App Logic
if faiss_file and csv_file:
    if 'models_and_data' not in st.session_state:
        st.session_state.models_and_data = load_models_and_data(faiss_file, csv_file)
    st.sidebar.success("Data loaded!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.retrieved_data = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a complex question about clinical trials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing query and searching..."):
                try:
                    inputs = {"question": prompt, "k": k_results, "current_batch": 0}
                    final_state = app_graph.invoke(inputs)
                    response = final_state.get("final_answer", "An error occurred.")
                    st.session_state.retrieved_data = final_state.get("retrieved_df")
                    st.markdown(response)
                except Exception as e:
                    st.error("An error occurred. The Groq API may be busy or an issue occurred with the request. Please try again.")
                    print(f"An error occurred: {e}")
        
        # This part needs to be inside the try block if you want it to run only on success
        if 'response' in locals() and response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Display full details and download button if data was retrieved
    if st.session_state.get("retrieved_data") is not None and not st.session_state.retrieved_data.empty:
        st.markdown("---")
        st.subheader("Retrieved Trial Details")
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv_data = convert_df_to_csv(st.session_state.retrieved_data)
        st.download_button(
            label="Download Full Results as CSV",
            data=csv_data,
            file_name="clinical_trial_results.csv",
            mime="text/csv",
        )

        for index, row in st.session_state.retrieved_data.iterrows():
            with st.expander(f"**{row.get('protocolSection.identificationModule.nctId', 'N/A')}**: {row.get('protocolSection.identificationModule.officialTitle', 'N/A')}"):
                st.dataframe(row)

else:
    st.info("Please upload your data files in the sidebar to begin.")

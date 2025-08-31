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
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=GROQ_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Groq API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
    
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'llama3-8b-8192'

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

# --- 3. THE UNRESTRICTED RETRIEVER TOOL ---
def search_clinical_trials(query: str, filters: dict):
    """
    Performs an exhaustive search to find ALL matching trials, returning them sorted by relevance.
    """
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

    # If there's a semantic query, rank the ENTIRE filtered set by relevance
    if query:
        # Get the original indices and vectors for the filtered items
        filtered_indices = working_df.index.to_numpy()
        # We need the original embeddings to re-rank. This assumes they are accessible or stored.
        # For simplicity here, we search the full index and then filter.
        query_vector = embedding_model.encode([query]).astype('float32')
        
        # Search the entire index to get a full relevance ranking
        distances, all_indices = search_index.search(query_vector, k=search_index.ntotal)
        
        # Intersect the fully ranked list with our filtered list to get a ranked, filtered list
        ranked_filtered_indices = [idx for idx in all_indices[0] if idx in filtered_indices]
        
        results_df = metadata_df.iloc[ranked_filtered_indices]
    else:
        # If no semantic query, return all filtered results
        results_df = working_df
        
    return results_df

# --- 4. LANGGRAPH AGENT SETUP ---
class AgentState(TypedDict):
    question: str
    k_summarize: int # How many to summarize in detail
    plan: dict
    retrieved_df: Optional[pd.DataFrame]
    total_found: int
    batch_summaries: List[str]
    final_answer: str
    current_batch: int

# --- Agent Nodes ---
def planner_node(state: AgentState):
    """Decomposes the user query."""
    prompt = f"""You are a query analysis expert. Your job is to break down a user's question about clinical trials into a semantic search query and structured filters.

    Identify filters like: sponsor, status, phase. The rest is the semantic query.

    User Question: "{state['question']}"

    Provide your answer ONLY as a valid JSON object with keys "query" and "filters".
    Example: {{"query": "lung cancer", "filters": {{"sponsor": "pfizer", "status": "recruiting"}}}}
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=LLM_MODEL_NAME, temperature=0, response_format={"type": "json_object"},
    )
    plan = json.loads(chat_completion.choices[0].message.content)
    return {"plan": plan}

def retrieve_node(state: AgentState):
    """Retrieves ALL documents based on the plan."""
    results_df = search_clinical_trials(
        query=state["plan"].get("query", ""),
        filters=state["plan"].get("filters", {})
    )
    return {"retrieved_df": results_df, "total_found": len(results_df)}

def batch_summarize_node(state: AgentState):
    """Summarizes one batch of the TOP k_summarize results."""
    df_to_summarize = state["retrieved_df"].head(state["k_summarize"])
    start_index = state["current_batch"] * 5
    end_index = start_index + 5
    batch_df = df_to_summarize.iloc[start_index:end_index]
    
    if batch_df.empty:
        return {"batch_summaries": state.get("batch_summaries", [])}

    context = "\n---\n".join([f"Trial {start_index + i + 1}:\n{row.to_string()}" for i, row in batch_df.iterrows()])
    prompt = f"Summarize the key findings from the following batch of clinical trials:\n\n{context}"
    
    chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=LLM_MODEL_NAME)
    summary = chat_completion.choices[0].message.content
    
    new_summaries = state.get("batch_summaries", []) + [summary]
    return {"batch_summaries": new_summaries, "current_batch": state["current_batch"] + 1}

def final_summary_node(state: AgentState):
    """Creates a final summary that reports the total count and summarizes the top results."""
    if not state.get("batch_summaries"):
        return {"final_answer": "No relevant trials were found based on your query."}
        
    context = "\n\n".join(state["batch_summaries"])
    prompt = f"""You are a professional clinical research assistant.
    
    First, state the TOTAL number of trials found, which is {state['total_found']}.
    
    Then, synthesize the following batch summaries into a single, cohesive final answer for the user's original question. Explain the key themes and provide details for the most relevant trials discussed.

    User Question: "{state['question']}"

    Summaries of the top {state['k_summarize']} trials:
    {context}
    """
    chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=LLM_MODEL_NAME)
    final_summary = chat_completion.choices[0].message.content
    return {"final_answer": final_summary}

# --- Conditional Edges ---
def should_summarize(state: AgentState):
    return "end" if state["retrieved_df"] is None or state["retrieved_df"].empty else "summarize"

def more_batches(state: AgentState):
    if state["current_batch"] * 5 >= state["k_summarize"] or state["current_batch"] * 5 >= state["total_found"]:
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
workflow.add_conditional_edges("retriever", should_summarize, {"end": END, "summarize": "batch_summarizer"})
workflow.add_conditional_edges("batch_summarizer", more_batches, {"to_final_summary": "final_summarizer", "continue_batching": "batch_summarizer"})
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
k_summarize_results = st.sidebar.slider("Number of top trials to summarize:", 1, 50, 10)

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
            with st.spinner("Performing comprehensive search and analysis..."):
                try:
                    inputs = {"question": prompt, "k_summarize": k_summarize_results, "current_batch": 0}
                    final_state = app_graph.invoke(inputs)
                    response = final_state.get("final_answer", "An error occurred.")
                    st.session_state.retrieved_data = final_state.get("retrieved_df")
                    st.markdown(response)
                except Exception as e:
                    st.error("An error occurred. The Groq API may be busy or an issue occurred with the request. Please try again.")
                    print(f"An error occurred: {e}")
        
        if 'response' in locals() and response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Display full details and download button if data was retrieved
    if st.session_state.get("retrieved_data") is not None and not st.session_state.retrieved_data.empty:
        st.markdown("---")
        st.subheader(f"Retrieved Trial Details (Top {k_summarize_results} of {len(st.session_state.retrieved_data)} shown)")
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv_data = convert_df_to_csv(st.session_state.retrieved_data)
        st.download_button(
            label=f"Download All {len(st.session_state.retrieved_data)} Results as CSV",
            data=csv_data,
            file_name="clinical_trial_results.csv",
            mime="text/csv",
        )

        # Only show expanders for the top k summarized results to keep the UI fast
        for index, row in st.session_state.retrieved_data.head(k_summarize_results).iterrows():
            with st.expander(f"**{row.get('protocolSection.identificationModule.nctId', 'N/A')}**: {row.get('protocolSection.identificationModule.officialTitle', 'N/A')}"):
                st.dataframe(row)
else:
    st.info("Please upload your data files in the sidebar to begin.")

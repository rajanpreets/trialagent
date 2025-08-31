import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# --- 1. CONFIGURATION ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Google API Key not found. Please add it to your Streamlit secrets.")
    st.stop()
    
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATIVE_MODEL_NAME = 'gemini-2.5-pro'

# --- 2. DATA AND MODEL LOADING ---
@st.cache_resource
def load_models_and_data(faiss_file_obj, metadata_file_obj):
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
    embedding_model, search_index, metadata_df = st.session_state.models_and_data
    
    # Start with the full dataframe
    working_df = metadata_df

    # Apply structured filters first
    if filters:
        for column, value in filters.items():
            # Find the actual column name that contains the filter key
            actual_col = next((c for c in metadata_df.columns if column in c.lower()), None)
            if actual_col and value:
                working_df = working_df[working_df[actual_col].str.contains(value, case=False, na=False)]

    if working_df.empty:
        return pd.DataFrame()

    # If there's a semantic query, perform vector search on the filtered results
    if query:
        # Get the original indices of the filtered rows
        filtered_indices = working_df.index.to_numpy()
        query_vector = embedding_model.encode([query]).astype('float32')
        
        # Retrieve a large number of initial candidates to ensure good overlap
        distances, initial_indices = search_index.search(query_vector, k=300)
        
        # Find the intersection between semantic results and filtered results
        final_indices = [idx for idx in initial_indices[0] if idx in filtered_indices]
        
        # Get the top k_final results
        results_df = metadata_df.iloc[final_indices[:k_final]]
    else:
        # If no semantic query, just return the top filtered results
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
    """Decomposes the user query into a semantic component and structured filters."""
    prompt = f"""You are a query analysis expert. Your job is to break down a user's question about clinical trials into a semantic search query and structured filters.

    Identify the following filters if they exist:
    - sponsor (e.g., pfizer, moderna)
    - status (e.g., recruiting, completed)
    - phase (e.g., phase 1, phase 2)

    The remaining part of the query is the semantic search query.

    User Question: "{state['question']}"

    Provide your answer as a JSON object with two keys: "query" and "filters". The "filters" value should be another JSON object.

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
    model = genai.GenerativeModel('gemini-1.5-pro-latest') # Use a fast model for planning
    response = model.generate_content(prompt)
    plan = json.loads(response.text.strip('`json\n'))
    return {"plan": plan}

def retrieve_node(state: AgentState):
    """Retrieves the full set of documents based on the plan."""
    results_df = search_clinical_trials(
        query=state["plan"].get("query"),
        filters=state["plan"].get("filters"),
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
    
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    
    new_summaries = state.get("batch_summaries", []) + [response.text]
    return {"batch_summaries": new_summaries, "current_batch": state["current_batch"] + 1}

def final_summary_node(state: AgentState):
    """Creates a final summary from all batch summaries."""
    if not state.get("batch_summaries"):
        return {"final_answer": "No relevant trials were found."}
        
    context = "\n\n".join(state["batch_summaries"])
    prompt = f"""You are a research assistant. The following are summaries from batches of clinical trials. Synthesize them into a single, cohesive final answer for the user's original question.

    User Question: "{state['question']}"

    Batch Summaries:
    {context}
    """
    model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    response = model.generate_content(prompt)
    return {"final_answer": response.text}

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
    {
        "end": END,
        "start_batch_summary": "batch_summarizer",
        "generate_direct_summary": "batch_summarizer" # Re-use batch node for direct summary
    }
)
workflow.add_conditional_edges(
    "batch_summarizer",
    more_batches,
    {
        "to_final_summary": "final_summarizer",
        "continue_batching": "batch_summarizer" # Loop
    }
)
workflow.add_edge("final_summarizer", END)
app_graph = workflow.compile()

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("Advanced Clinical Trials Research Assistant 🔬")

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
                inputs = {"question": prompt, "k": k_results, "current_batch": 0}
                final_state = app_graph.invoke(inputs)
                response = final_state.get("final_answer", "An error occurred.")
                st.session_state.retrieved_data = final_state.get("retrieved_df")
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun() # Rerun to display expanders/download button immediately

    # Display full details and download button if data was retrieved
    if st.session_state.get("retrieved_data") is not None and not st.session_state.retrieved_data.empty:
        st.markdown("---")
        st.subheader("Retrieved Trial Details")
        
        # Convert DataFrame to CSV for download
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

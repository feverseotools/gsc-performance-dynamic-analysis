import streamlit as st
import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

####################################
# 1. CUSTOM ANALYSIS FUNCTIONS     #
####################################
def top_queries_by_clicks(df, n=10):
    """
    Returns the top 'n' queries based on total clicks.
    Assumes the DataFrame has columns 'query' and 'clicks'.
    """
    return (
        df.groupby("query", as_index=False)["clicks"]
        .sum()
        .sort_values("clicks", ascending=False)
        .head(n)
    )

def avg_ctr_by_page(df):
    """
    Returns the average CTR by page.
    Assumes the DataFrame has columns 'page' and 'ctr'.
    """
    return (
        df.groupby("page", as_index=False)["ctr"]
        .mean()
        .sort_values("ctr", ascending=False)
    )

def filter_by_country(df, country_code):
    """
    Filters the DataFrame by a given country code (e.g., 'US', 'ES').
    Assumes the DataFrame has a column 'country'.
    """
    return df[df["country"] == country_code].copy()

def summarize_data(df):
    """
    Returns a descriptive summary of the DataFrame.
    """
    return df.describe(include="all")


#######################################
# 2. OPTIONAL: EMBEDDINGS FOR QUERIES #
#######################################
def compute_embeddings_for_queries(df, query_col="query"):
    """
    Demonstration of how to call OpenAI Embeddings for each 'query' in the DataFrame.
    NOTE: This can be expensive if the DataFrame is large.
    """
    queries = df[query_col].unique().tolist()
    
    query_embeddings = {}
    for q in queries:
        emb_resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=q
        )
        embedding = emb_resp["data"][0]["embedding"]
        query_embeddings[q] = embedding
    
    return query_embeddings

def semantic_search(user_query, query_embeddings, top_k=5):
    """
    Performs a simple semantic search using the embedding of `user_query`
    against previously computed embeddings.
    Returns the top_k most similar queries and their similarity scores.
    """
    # Embed the user query
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=user_query
    )
    user_emb = np.array(response["data"][0]["embedding"]).reshape(1, -1)
    
    # Prepare all embeddings
    all_queries = list(query_embeddings.keys())
    all_vectors = np.array(list(query_embeddings.values()))
    
    # Compute cosine similarity
    sims = cosine_similarity(user_emb, all_vectors)[0]
    
    # Sort descending by similarity
    top_indices = sims.argsort()[::-1][:top_k]
    results = [(all_queries[i], float(sims[i])) for i in top_indices]
    return results


#############################################
# 3. GPT CODE GENERATION (CHAT COMPLETIONS) #
#############################################
def generate_python_code_from_gpt(messages, model="gpt-3.5-turbo", max_tokens=800):
    """
    Sends a list of messages to OpenAI ChatCompletion to generate Python code.
    Returns the content of the assistant's message.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]


####################
# 4. KEY VALIDATION #
####################
def test_openai_key(api_key):
    """
    Attempts a minimal call (listing available models) using the provided key
    to check if it's valid. Returns True if successful.
    """
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception:
        return False


###############################
# 5. STREAMLIT APP (MAIN CODE)
###############################
def main():
    """
    Main Streamlit application entry point.
    """

    ###############################
    # App Title and Introduction
    ###############################
    st.title("Advanced GSC Explorer with OpenAI")
    st.write("""
    This application allows you to:
    1. Provide an **OpenAI API key** and verify its validity.
    2. **Upload** Google Search Console data (CSV or Excel format).
    3. **Optionally filter** columns for a focused dataset.
    4. **Ask questions** to GPT about your data or request specific analyses.
    5. **(Optional)** Compute embeddings for semantic search on the `query` column.
    
    **How to Use:**
    - Enter your OpenAI API key below.
    - Once valid, upload your GSC file and explore the data.
    - Compute embeddings (optional) if you want to do **semantic:search**.
    - Type questions in the chat; GPT will generate Python code and run it on your data.
    """)

    ################################
    # 5.1 Input field for API Key  #
    ################################
    st.subheader("Step 1: Enter your OpenAI API Key")
    st.write("Please enter your OpenAI API key below. We will use it **only** during this session to connect to OpenAI services.")
    
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    st.session_state["api_key"] = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state["api_key"],
        placeholder="sk-XXXXXXXXXXXXXXXXXXXXXXXX"
    )
    
    # Button to check the key
    if st.button("Check Key"):
        if test_openai_key(st.session_state["api_key"]):
            st.success("API Key is valid! You can proceed.")
        else:
            st.error("API Key is invalid or could not be verified. Please try again.")
    
    # Ensure we have a valid key before proceeding
    if not test_openai_key(st.session_state["api_key"]):
        st.warning("Please enter a **valid OpenAI API Key** to proceed with the data analysis.")
        return
    
    # Once the key is validated, set it globally
    openai.api_key = st.session_state["api_key"]
    
    ##########################################
    # 5.2 Upload GSC File (Excel or CSV)
    ##########################################
    st.subheader("Step 2: Upload Your GSC Data")
    st.write("Upload your GSC export file in **CSV** or **Excel** format. The application will parse it into a Pandas DataFrame.")
    
    uploaded_file = st.file_uploader("Upload GSC file (CSV or XLSX)", type=["xlsx", "csv"])
    
    # Session states for data
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "filtered_df" not in st.session_state:
        st.session_state["filtered_df"] = None
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Load and display the data
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state["df"] = df
        st.session_state["filtered_df"] = df
        
        st.write("**Data Preview:**")
        st.dataframe(df.head())
        
        #####################################
        # 5.2.1 (Optional) Column Filtering
        #####################################
        st.subheader("Step 3: (Optional) Column Filtering")
        st.write("Select which columns you want to keep. This can help you focus on specific parts of the dataset.")
        
        selected_cols = st.multiselect(
            "Columns to keep:",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
        
        if selected_cols:
            st.session_state["filtered_df"] = df[selected_cols]
            st.write("Filtered Data Preview:")
            st.dataframe(st.session_state["filtered_df"].head())
        
        ################################################
        # 5.2.2 (Optional) Compute Embeddings for Queries
        ################################################
        st.write("""
        **Step 4 (Optional):** Compute embeddings for the `query` column in your dataset.
        This enables a **semantic search** feature.  
        *Usage:* Type `"semantic: your text..."` in the chat box to find similar queries.
        """)
        if st.button("Compute Query Embeddings"):
            with st.spinner("Computing embeddings... This may take some time depending on your data size."):
                st.session_state["embeddings"] = compute_embeddings_for_queries(
                    st.session_state["filtered_df"],
                    query_col="query"  # Adapt if your column name is different
                )
            st.success("Embeddings computed and stored in memory!")

    ###################################
    # 5.3 Chat Interface with GPT
    ###################################
    st.subheader("Step 5: Ask GPT about Your Data")
    st.write("""
    Type your questions or instructions here.  
    - If you start your message with `"semantic:"`, the app will perform a semantic search using embeddings (if computed).  
    - Otherwise, GPT will generate Python code to analyze `df` or `filtered_df`.  
    - The generated code is displayed and then executed. If a `result` or `fig` is created, it will be shown below.
    """)
    
    user_input = st.chat_input("Enter your query or command...")

    if user_input:
        # Save user message in history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        
        ########################################
        # Semantic Search Trigger
        ########################################
        if user_input.strip().lower().startswith("semantic:"):
            if st.session_state["embeddings"] is None:
                st.chat_message("assistant").write(
                    "Embeddings are not computed yet. Please compute them first (Step 4)."
                )
            else:
                query_text = user_input.replace("semantic:", "").strip()
                results = semantic_search(query_text, st.session_state["embeddings"], top_k=5)
                
                msg = f"**Semantic Search Results** for '{query_text}':\n\n"
                for r, sim in results:
                    msg += f"- {r} (similarity={sim:.4f})\n"
                
                # Add to chat history and display
                st.session_state["chat_history"].append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
        
        ########################################
        # Otherwise, Generate Python Code with GPT
        ########################################
        else:
            # Prepare system instructions for GPT
            system_message = (
                "You are an assistant that generates only Python code to analyze a Pandas DataFrame named "
                "'df' or 'st.session_state.filtered_df' containing Google Search Console data. "
                "You also have access to these custom functions:\n\n"
                "1) top_queries_by_clicks(df, n=10)\n"
                "2) avg_ctr_by_page(df)\n"
                "3) filter_by_country(df, country_code)\n"
                "4) summarize_data(df)\n\n"
                "Use them as needed or generate your own code. If you generate plots with matplotlib, "
                "assign the figure to 'fig'. The final output must be stored in a 'result' variable (or 'fig').\n\n"
                "Do not provide explanations. Return Python code only."
            )
            
            messages = [
                {"role": "system", "content": system_message},
                # We only append the latest user input to keep it simple;
                # you could include more context if desired.
                {"role": "user", "content": user_input}
            ]
            
            code_reply = generate_python_code_from_gpt(messages=messages)
            
            # Display GPT's code
            st.chat_message("assistant").write("**GPT Generated Code:**")
            st.code(code_reply, language="python")
            
            # Attempt to execute the generated code
            try:
                local_env = {
                    "pd": pd,
                    "df": st.session_state["df"],
                    "filtered_df": st.session_state["filtered_df"],
                    "result": None,
                    "fig": None,
                    "top_queries_by_clicks": top_queries_by_clicks,
                    "avg_ctr_by_page": avg_ctr_by_page,
                    "filter_by_country": filter_by_country,
                    "summarize_data": summarize_data,
                    "np": np
                }
                
                exec(code_reply, {}, local_env)
                
                # Check for 'result'
                if local_env.get("result") is not None:
                    st.chat_message("assistant").write("**Result:**")
                    result = local_env["result"]
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    else:
                        st.write(result)
                
                # Check for 'fig'
                if local_env.get("fig") is not None:
                    fig = local_env["fig"]
                    st.chat_message("assistant").write("**Plot:**")
                    st.pyplot(fig)
            
            except Exception as e:
                error_msg = f"**Error executing GPT-generated code:**\n{e}"
                st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").write(error_msg)
    
    ###############################################
    # Display the entire chat history at the end
    ###############################################
    if st.session_state["chat_history"]:
        st.write("### Conversation History")
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])


if __name__ == "__main__":
    main()

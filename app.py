import streamlit as st
import pandas as pd
import openai
import os

##########################################
# 1. Cost & Token Tracking Configuration #
##########################################
# These rates apply to gpt-3.5-turbo as an example:
GPT_35_TURBO_PROMPT_RATE = 0.0015  # USD per 1K prompt tokens
GPT_35_TURBO_COMPLETION_RATE = 0.0020  # USD per 1K completion tokens

def calculate_cost_from_usage(usage, model="gpt-3.5-turbo"):
    """
    Calculates the cost of an OpenAI API call based on token usage.
    usage is typically: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    """
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    
    if model == "gpt-3.5-turbo":
        cost_input = (prompt_tokens / 1000.0) * GPT_35_TURBO_PROMPT_RATE
        cost_output = (completion_tokens / 1000.0) * GPT_35_TURBO_COMPLETION_RATE
        cost_total = cost_input + cost_output
        return cost_total
    else:
        # Fallback or handle other models
        return 0.0

#####################################
# 2. Simple Key Validation Function #
#####################################
def test_openai_key(api_key):
    """
    Attempts a minimal call (listing models) using the provided key.
    Returns True if it works, otherwise False.
    """
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception:
        return False

############################################
# 3. GPT Code Generation with Usage Return #
############################################
def generate_python_code_from_gpt(prompt):
    """
    Sends a prompt to GPT (ChatCompletion) and returns (code, usage).
    """
    messages = [
        {"role": "system", "content": "You are a Python code generator."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=400,
        )
        code = response["choices"][0]["message"]["content"]
        usage = response["usage"]  # includes prompt_tokens, completion_tokens, total_tokens
        return code, usage
    except Exception as e:
        error_text = f"# Error generating code: {e}"
        return error_text, None

#################################
# 4. Streamlit Main Application #
#################################
def main():
    # Optionally configure page layout
    st.set_page_config(page_title="GSC + Cost Tracker", layout="wide")

    st.title("Google Search Console Explorer (with OpenAI Cost Tracker)")
    st.write("""
    This demo shows how to:
    1. Put all steps into a **left sidebar** (using `st.sidebar.expander`).
    2. Track **token usage** and **cost** for each GPT request.
    3. Accumulate a **running total** of tokens and cost in `st.session_state`.
    """)

    # Initialize session states
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "code_generated" not in st.session_state:
        st.session_state["code_generated"] = ""
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "total_tokens" not in st.session_state:
        st.session_state["total_tokens"] = 0
    if "total_cost" not in st.session_state:
        st.session_state["total_cost"] = 0.0

    ########################
    # SIDEBAR & EXPANDERS
    ########################
    with st.sidebar.expander("Step 1: Provide OpenAI API Key", expanded=True):
        st.write("Enter your OpenAI API key here and check its validity:")
        st.session_state["api_key"] = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state["api_key"],
            placeholder="sk-XXXXXXXXXXXXXXXXXXXX"
        )
        if st.button("Check Key"):
            if test_openai_key(st.session_state["api_key"]):
                st.success("API Key is valid!")
            else:
                st.error("Invalid API Key or network issue.")

    # Validate key before proceeding
    if not test_openai_key(st.session_state["api_key"]):
        st.warning("Enter a valid OpenAI API key to continue.")
        return
    else:
        openai.api_key = st.session_state["api_key"]

    with st.sidebar.expander("Step 2: Upload Your GSC File", expanded=True):
        st.write("Upload a CSV or Excel file. The data will appear in the main area.")
        uploaded_file = st.file_uploader("Select CSV or XLSX", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                st.session_state["df"] = pd.read_csv(uploaded_file)
            else:
                st.session_state["df"] = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")

    with st.sidebar.expander("Step 3: Ask GPT for Code", expanded=True):
        st.write("Type a request about the data. GPT will generate Python code.")
        user_prompt = st.text_area(
            "Your instruction to GPT about 'df':",
            help="Example: 'Show the top 5 rows' or 'Calculate basic stats.'"
        )

        if st.button("Generate & Run Code"):
            if st.session_state["df"] is not None:
                if user_prompt.strip():
                    code, usage = generate_python_code_from_gpt(
                        f"You have a Pandas DataFrame named 'df'. {user_prompt}"
                    )
                    st.session_state["code_generated"] = code

                    # Track usage & cost
                    if usage is not None:
                        # Calculate cost from usage
                        call_cost = calculate_cost_from_usage(usage, model="gpt-3.5-turbo")
                        st.session_state["total_tokens"] += usage["total_tokens"]
                        st.session_state["total_cost"] += call_cost
                else:
                    st.warning("Please enter a valid prompt.")
            else:
                st.warning("You need to upload a file first.")

    ##################################
    # MAIN PAGE: Display Results
    ##################################
    st.write("## Main Page Output")

    # 1) Show Accumulated Cost
    st.write(f"**Total Tokens Used:** {st.session_state['total_tokens']}")
    st.write(f"**Estimated Total Cost (USD):** ${st.session_state['total_cost']:.4f}")

    # 2) Show Data Preview (if df is uploaded)
    if st.session_state["df"] is not None:
        with st.expander("Data Preview", expanded=True):
            st.write("Here are the first rows of your uploaded data:")
            st.dataframe(st.session_state["df"].head())

    # 3) Show GPT-Generated Code
    if st.session_state["code_generated"]:
        with st.expander("GPT-Generated Code", expanded=True):
            st.write("This is the code GPT generated based on your prompt:")
            st.code(st.session_state["code_generated"], language="python")
            
            # Attempt to run code if generated
            local_env = {"df": st.session_state["df"], "result": None}
            try:
                exec(st.session_state["code_generated"], {}, local_env)
                st.session_state["result"] = local_env.get("result", None)
            except Exception as e:
                st.error(f"Error executing code: {e}")

    # 4) Show Execution Result
    if st.session_state["result"] is not None:
        with st.expander("Execution Result", expanded=True):
            st.write("The code's 'result' variable is shown below:")
            result = st.session_state["result"]
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            else:
                st.write(result)


if __name__ == "__main__":
    main()

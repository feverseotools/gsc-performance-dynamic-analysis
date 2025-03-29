import streamlit as st
import pandas as pd
import openai
import os

###############################
# 1. Simple Key Validation
###############################
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


###############################
# 2. GPT Code Generation
###############################
def generate_python_code_from_gpt(prompt):
    """
    Sends a simple prompt to GPT (ChatCompletion) and returns generated Python code.
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
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"# Error generating code: {e}"


###############################
# 3. Streamlit App
###############################
def main():
    # Optional: You can set wide layout for more horizontal space.
    st.set_page_config(page_title="GSC Explorer", layout="wide")

    st.title("GSC Explorer with Sidebar & Expanders")
    st.write("""
    This example application shows how to place all steps and configurations
    in a **left sidebar**, with each section in a **collapsible expander**.
    The resulting actions (like data previews, generated code, and execution
    outputs) appear in the main page area.
    """)

    # Session state for storing the DataFrame and API key
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "code_generated" not in st.session_state:
        st.session_state["code_generated"] = ""
    if "result" not in st.session_state:
        st.session_state["result"] = None

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
        # Set the key globally
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
        st.write("Type a request or question about the data. GPT will generate Python code.")
        user_prompt = st.text_area(
            "Your instruction to GPT about 'df':",
            help="For example: 'Show the top 5 rows' or 'Calculate basic stats.'"
        )

        if st.button("Generate & Run Code"):
            if not st.session_state["df"] is None:
                if user_prompt.strip():
                    # Generate code from GPT
                    code = generate_python_code_from_gpt(
                        f"You have a Pandas DataFrame named 'df'. {user_prompt}"
                    )
                    st.session_state["code_generated"] = code
                else:
                    st.warning("Please enter a valid prompt.")
            else:
                st.warning("You need to upload a file first.")


    ##################################
    # MAIN PAGE: Display Results
    ##################################
    st.write("## Main Page Output")

    # 1) Show Data Preview (if df is uploaded)
    if st.session_state["df"] is not None:
        with st.expander("Data Preview", expanded=True):
            st.write("Here are the first rows of your uploaded data:")
            st.dataframe(st.session_state["df"].head())

    # 2) Show GPT-Generated Code
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

    # 3) Show Execution Result
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

import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.langchain import LangchainLLM
from langchain_openai import ChatOpenAI
import time

# Set page config with dark theme
st.set_page_config(
    page_title="SambaNova Multimodal Assistant",
    layout="wide"
)

# Apply custom CSS for dark theme
st.markdown(
    """
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #2E2E2E;
            color: white;
        }
        .stButton>button {
            background-color: #444;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #FBBF24;
            color: black;
        }
        .sidebar .sidebar-content {
            background-color: #1E1E1E;
        }
        .stTextArea>div>textarea {
            background-color: #2E2E2E;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display Title and Subtitle in the center
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>
        <span style='color: #FBBF24;'>SambaNova</span> data analysis Assistant
    </h1>
    <h3 style='text-align: center; color: #A0A0A0;'>
        Mid-Senior Level Data Analysis Agent using SambaNova API
    </h3>
    """,
    unsafe_allow_html=True
)

# Sidebar for Setup
st.sidebar.title("Setup")
st.sidebar.markdown("Get your SambaNova API key [here](https://www.sambanova.ai/)")

# Input for API Key
sambanova_api_key = st.sidebar.text_input(
    "SAMBANOVA CLOUD API KEY", 
    type="password"
)

# Save Credentials Button
if st.sidebar.button("Save Credentials"):
    if sambanova_api_key:
        st.sidebar.success("API Key Saved Successfully")
    else:
        st.sidebar.error("Please enter an API Key")

# Function to authenticate SambaNova API using Langchain
@st.cache_resource
def authenticate_sambanova(api_key):
    return ChatOpenAI(
        base_url="https://api.sambanova.ai/v1/",
        api_key=api_key,
        streaming=False,
        model="DeepSeek-R1-Distill-Llama-70B",
    )

# Authenticate API Key
if sambanova_api_key:
    try:
        sambanova_llm = authenticate_sambanova(sambanova_api_key)
        langchain_llm = LangchainLLM(sambanova_llm)
        st.sidebar.success("Authentication Successful")
    except Exception as e:
        st.sidebar.error(f"Authentication failed: {e}")
        st.stop()
else:
    st.sidebar.info("Enter API key to proceed")
    st.stop()

# Sidebar for file upload
st.sidebar.title("Upload Data Files")
file_type = st.sidebar.radio("Select file type", ("CSV", "Excel"))
uploaded_file_1 = st.sidebar.file_uploader(f"Upload the first {file_type} file", type=["csv", "xls", "xlsx"])
uploaded_file_2 = st.sidebar.file_uploader(f"Upload the second {file_type} file (optional)", type=["csv", "xls", "xlsx"])

# Main application content
if uploaded_file_1 is not None:
    try:
        # Load data
        if file_type == "CSV":
            data1 = pd.read_csv(uploaded_file_1)
        else:
            data1 = pd.read_excel(uploaded_file_1, engine="openpyxl")

        # Display first dataset preview
        st.subheader("First Dataset Preview")
        st.write(data1.head())

        # Display general information
        st.subheader("General Information (First Dataset)")
        st.write(f"Shape: {data1.shape}")
        st.write(f"Data Types:\n{data1.dtypes}")
        st.write(f"Memory Usage: {data1.memory_usage(deep=True).sum()} bytes")

        # Initialize SmartDataframe with SambaNova LLM via PandasAI
        df_smart1 = SmartDataframe(data1, config={"llm": langchain_llm})

        # Check if second file is uploaded
        if uploaded_file_2 is not None:
            if file_type == "CSV":
                data2 = pd.read_csv(uploaded_file_2)
            else:
                data2 = pd.read_excel(uploaded_file_2, engine="openpyxl")

            # Display second dataset preview
            st.subheader("Second Dataset Preview")
            st.write(data2.head())

            # Display general information
            st.subheader("General Information (Second Dataset)")
            st.write(f"Shape: {data2.shape}")
            st.write(f"Data Types:\n{data2.dtypes}")
            st.write(f"Memory Usage: {data2.memory_usage(deep=True).sum()} bytes")

            # Merge both datasets
            st.subheader("Merged Dataset Check")
            common_columns = list(set(data1.columns) & set(data2.columns))
            
            if common_columns:
                merged_data = pd.merge(data1, data2, on=common_columns, how="inner")
                st.write(merged_data.head())

                # Display merged dataset general information
                st.write(f"Shape of Merged Dataset: {merged_data.shape}")
                
                # Initialize SmartDataframe with merged data
                df_smart = SmartDataframe(merged_data, config={"llm": langchain_llm})
            else:
                st.warning("No common columns found. Unable to merge datasets.")

        else:
            df_smart = df_smart1  # Only one dataset is available

        # Chat-style Input for Natural Language Query
        query = st.text_input(
            "Ask questions about your data",
            placeholder="Example: 'Show me the average sales per month' or 'Check the correlation between both datasets'"
        )

        if query:
            try:
                # Query processing
                start_time = time.time()
                response = df_smart.chat(query)
                end_time = time.time()

                # Display response
                st.subheader("AI Response")
                st.write(response)
                st.success(f"Query processed in {end_time - start_time:.2f} seconds.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer
st.markdown(
    """
    <footer style='text-align: center; padding: 10px; background-color: #1E1E1E; color: white;'>
       Powered by SambaNova API, PandasAI, and Open Source Tools<br>
       Made with ❤️ by Martin Khristi
    </footer>
    """,
    unsafe_allow_html=True,
)
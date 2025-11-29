
import os
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import speech_recognition as sr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

api_key = "AIzaSyDaM0twsGt6Rv5M2pY4ze4lZlKc7IQWuiQ"
llm_name = "gemini-2.5-flash"
model = ChatGoogleGenerativeAI(api_key=api_key, model=llm_name)

# database
db_dir = "./db"
os.makedirs(db_dir, exist_ok=True)
database_file_path = os.path.join(db_dir, "uploaded_files.db")
engine = create_engine(f"sqlite:///{database_file_path}")

db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

MSSQL_AGENT_PREFIX = """
You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown, but **do not include markdown backticks in Action Input.**
- ALWAYS include a section starting with "Explanation:" showing how you arrived at the answer with the SQL query used.
- If unrelated to the database, respond with "I don't know".
"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """
## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.
"""

sql_agent = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=model,
    toolkit=toolkit,
    top_k=30,
    verbose=True,
)

st.set_page_config(page_title="SQL Query AI Agent", layout="wide")
st.title("SQL Query AI Agent with Voice")
st.markdown("Upload CSV files to build your database, then query using natural language or voice input.")


uploaded_file = st.file_uploader("Upload a CSV file to add to the database:", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file).fillna(0)
        table_name = uploaded_file.name.split(".")[0]
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        st.success(f"File '{uploaded_file.name}' uploaded and table '{table_name}' created successfully!")
        st.write(df.head())
    except Exception as e:
        st.error(f"Failed to upload or parse file: {e}")

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak your query.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing voice input...")
            query = recognizer.recognize_google(audio)
            st.success(f"Recognized: {query}")
            return query.strip()
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Voice service error: {e}")
        except Exception as e:
            st.error(f"Voice capture error: {e}")
    return None


st.sidebar.header("Query Input Options")
input_mode = st.sidebar.radio("Choose input mode:", ["Text", "Voice"])

if "question" not in st.session_state:
    st.session_state["question"] = ""

if input_mode == "Text":
    st.session_state["question"] = st.text_input("Enter your natural language SQL query:", value=st.session_state["question"])
else:
    if st.button("Record Voice Query"):
        voice_query = get_voice_input()
        if voice_query:
            st.session_state["question"] = voice_query

st.write(f"**Current Query:** {st.session_state['question']}")


if st.button("Run Query"):
    if st.session_state["question"]:
        try:
            st.info(f"Running: {st.session_state['question']}")
            res = sql_agent.invoke(st.session_state["question"])
            if "output" in res:
                st.markdown(res["output"])
            else:
                st.error("The agent did not return an output.")
        except Exception as e:
            st.error(f"Error executing query: {e}")
    else:
        st.warning("Please enter a query before running.")


import os
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import speech_recognition as sr


api_key ="AIzaSyDaM0twsGt6Rv5M2pY4ze4lZlKc7IQWuiQ"

llm_name = "gemini-2.5-flash"
model = ChatGoogleGenerativeAI(api_key=api_key, model=llm_name)

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# Path to SQLite database
database_file_path = "./db/salary.db"

engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "./salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)



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
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
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

db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

sql_agent = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=model,
    toolkit=toolkit,
    top_k=30,
    verbose=True,
)

st.set_page_config(
    page_title="SQL Query AI Agent",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("Voice-SQL AI Agent")
st.markdown(
    """
    Welcome to the **SQL Query AI Agent**! This app allows you to:
    - Query a database using natural language.
    - Use voice input for queries.
    """
)

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your query... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing your voice input...")
            query = recognizer.recognize_google(audio)
            st.success(f"Recognized query: {query}")
            return query.strip()  # Strip extra spaces
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return None

if "question" not in st.session_state:
    st.session_state["question"] = ""

st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input mode:", ("Text", "Voice"))

if input_mode == "Text":
    st.session_state["question"] = st.text_input("Enter your query:", value=st.session_state["question"])
elif input_mode == "Voice":
    if st.button("Record Voice Query"):
        voice_query = get_voice_input()
        if voice_query:
            st.session_state["question"] = voice_query

st.write(f"Current query: {st.session_state['question']}")

if st.button("Run Query"):
    if st.session_state["question"]:
        try:
            st.info(f"Processing query: {st.session_state['question']}")

            res = sql_agent.invoke(st.session_state["question"])

            st.info("Query executed successfully. Raw output:")

            if "output" in res:
                st.markdown(res["output"])
            else:
                st.error("No output returned by the SQL agent.")
        except Exception as e:
            st.error(f"An error occurred while processing the query: {e}")
    else:
        st.error("Please provide a query.")


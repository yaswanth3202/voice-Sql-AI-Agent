# Voice SQL AI Agent

A Streamlit-based application that allows users to upload CSV files, build a SQL database, and query it using natural language or voice input. Powered by Google Gemini LLM and LangChain SQL tools, the agent automatically converts natural language questions into SQL queries and returns results.

---

## Features

- **CSV Upload:** Upload one or multiple CSV files to create tables in an SQLite database.
- **Natural Language Querying:** Ask questions in plain English and get SQL query results.
- **Voice Input:** Speak your query, and the system will recognize and process it automatically.
- **SQL Agent Powered by LLM:** Uses Google Gemini LLM to generate syntactically correct SQL queries.
- **Real-time Query Results:** Get answers directly from the database, with explanations of the queries executed.
- **Session Persistence:** Maintains current question across interactions in Streamlit.

---

## Installation

1. Clone the repository:

```bash
git clone "https://github.com/vinith-369/Voice-SQL-AI-Agent.git"
cd Voice-SQL-AI-Agent

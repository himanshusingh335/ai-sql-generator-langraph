from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from IPython.display import Image, display
from typing import Annotated
from pydantic import BaseModel
from typing_extensions import TypedDict
import os
from datetime import date
import sqlite3
import json


#--------define LLM---------
llm=init_chat_model("openai:gpt-4.1-mini")
#--------define state--------
class CustomState(AgentState):
    #messages: Annotated[list, add_messages]
    #remaining_steps: int
    query: str

#--------define tools--------

def get_todays_date() -> str:
    "This tool will get today's date in YYYY-MM-DD format."
    print("[DEBUG] get_todays_date tool called")
    today = date.today().strftime("%Y-%m-%d")  # e.g., "2025-08-14"
    return today

def execute_sqlite_select(query: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Executes a SELECT query on the SQLite database located at data/budget.db.
    Returns results as a string. Only SELECT queries are allowed.
    """
    try:
        # Enforce SELECT-only rule
        if not query.strip().lower().startswith("select"):
            return "Error: Only SELECT queries are allowed."

        conn = sqlite3.connect("data/budget.db")
        cursor = conn.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description]
        conn.close()

        # Format results
        results = [dict(zip(col_names, row)) for row in rows]
        state_update = {
        "query": query,   
        "messages": [ToolMessage(str(results), tool_call_id=tool_call_id)],
    }
        return Command(update=state_update)

    except Exception as e:
        state_update = {
        "query": query,   
        "messages": [ToolMessage(f"Error executing query: {e}", tool_call_id=tool_call_id)],
    }
        return Command(update=state_update)

def inspect_sqlite_db():
    """
    Returns all table names, their schema (CREATE TABLE statement),
    and the first 5 rows from each table in the SQLite database (data/data.db).
    Output is JSON.
    """
    db_path = "data/budget.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Step 1: Get all table names
        print("[DEBUG] Inspecting SQLite database at", db_path)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        
        db_info = {}
        
        for table in tables:
            # Step 2: Get schema
            print(f"[DEBUG] Inspecting table: {table}")
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
            schema = cursor.fetchone()["sql"]
            
            # Step 3: Get first 5 rows
            print(f"[DEBUG] Fetching sample rows from table: {table}")
            cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
            rows = [dict(r) for r in cursor.fetchall()]
            
            db_info[table] = {
                "schema": schema,
                "sample_rows": rows
            }
        
        return json.dumps(db_info, indent=2)
    
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})
    
    finally:
        conn.close()

#--------add conversational memory---------
memory = InMemorySaver()

agent = create_react_agent(
    model=llm,
    state_schema=CustomState,
    checkpointer=memory,
    tools=[inspect_sqlite_db, get_todays_date, execute_sqlite_select],
    prompt="""
            Role:
    You are an expert in writing and optimizing SQLite queries.

    Task:
    When given a natural language question:
    	1.	Understand the Database:
    	•	Use the provided tools to retrieve and analyze the database schema.
    	•	Consider the structure, relationships, and data types when forming queries.
    	2.	Query Construction:
    	•	Based on the schema and the question, write one or more SQL queries that will return the information needed to answer the question.
    	•	If the question involves dates, you may use the provided tool to get today’s date.
    	3.	Validation:
    	•	Review and validate each query to ensure correctness and efficiency.
    	•	Confirm that the queries are syntactically correct and align with the schema.
    	4.	Execution & Answer:
    	•	Execute the validated SQL queries using the provided tool.
    	•	Interpret the query results and provide a clear, concise, and accurate answer to the original question. Use Rupees as currency.

    Important: Always rely on the provided tools for schema access, date retrieval, and query execution.
            """
)

config = {"configurable": {"thread_id": "1"}}

  
while True:
    user_input = input("\nEnter your question (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break


    # Invoke the agent with the provided system prompt already included in the agent
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_input}],
        "query": ""
    }, config=config)

    try:
        # Print query if it exists
        if isinstance(response, dict):
            if "query" in response and response["query"]:
                print("\nExecuted Query:", response["query"])
            
            # Extract the last message from the response
            if "messages" in response:
                messages = response["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        print("\nAI:", last_message.content)
                    else:
                        print("\nAI:", str(last_message))
            else:
                print("\nAI:", str(response))
    except Exception as e:
        print("\nError processing response:", str(e))
        print("Raw response:", str(response))
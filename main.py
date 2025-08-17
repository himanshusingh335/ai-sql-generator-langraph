from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from IPython.display import Image, display
from typing import Annotated
from typing_extensions import TypedDict
import os
from datetime import date
import sqlite3
import json


#--------define LLM---------
query_generator_llm = init_chat_model("openai:gpt-4.1-mini")
#--------define state--------
class State(MessagesState):
    query: list[str]

#--------define tools--------

query_list=[]

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

        query_list.append(query)

        # Format results
        results = [dict(zip(col_names, row)) for row in rows]
        state_update = {
        "query": query_list,   
        "messages": [ToolMessage(str(results), tool_call_id=tool_call_id)],
    }
        return Command(update=state_update)

    except Exception as e:
        state_update = {
        "query": query_list,   
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

tools=[get_todays_date, inspect_sqlite_db, execute_sqlite_select]
#bind the tools to the LLM
query_generator_llm_with_tools = query_generator_llm.bind_tools(tools)
#Create node for the tool
sql_generator_tool_node = ToolNode(tools=tools)

#--------add conversational memory---------
memory = InMemorySaver()

#
#--------define state---------
# Define the state of the chatbot using MessagesState as base and add extra fields.

graph_builder = StateGraph(State)

#--------define nodes--------

def query_generator(state: State):
    
    response = query_generator_llm_with_tools.invoke(state["messages"])
    # Ensure messages remain list of dicts, do not overwrite with LangChain objects
    return {"messages": [response]}

# Build graph with tool routing
graph_builder = StateGraph(State)

# Add LLM nodes
graph_builder.add_node("query_generator", query_generator)
graph_builder.add_node("tools", sql_generator_tool_node)

# Add edges with conditional routing
graph_builder.add_edge(START, "query_generator")

graph_builder.add_conditional_edges(
    "query_generator",
    tools_condition,
)

graph_builder.add_edge("tools", "query_generator")

graph_builder.add_edge("query_generator", END)

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
#--------display graph--------
try:
    img_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img_bytes)
    print("Graph image saved as graph.png in the current folder.")
except Exception:
    # This requires some extra dependencies and is optional
    pass

#--------system prompt--------
system_prompt = """
Role:
You are an expert in writing and optimizing SQLite queries.

Task:
When given a natural language question:
    1.	Understand the Database:
    •	Use the provided tools to retrieve and analyze the database schema.
    •	Consider the structure, relationships, and data types when forming queries.
    2.	Query Construction:
    •	Based on the schema and the question, write one or more SQL queries that will return the information needed to answer the question.
    •	If the question involves dates, you may use the provided tool to get today's date.
    3.	Validation:
    •	Review and validate each query to ensure correctness and efficiency.
    •	Confirm that the queries are syntactically correct and align with the schema.
    4.	Execution & Answer:
    •	Execute the validated SQL queries using the provided tool.
    •	Interpret the query results and provide a clear, concise, and accurate answer to the original question. Use Rupees as currency.

Important: Always rely on the provided tools for schema access, date retrieval, and query execution.
"""

#--------run the app--------
if __name__ == "__main__":
    try:
        # Initialize conversation with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        while True:
            user_question = input("Enter your question: ")
            
            # Add user question to messages
            messages.append({"role": "user", "content": user_question})
            
            # Create state for this turn
            current_state = {
                "messages": messages.copy(),
                "query": []
            }

            print("\n--- Streaming execution ---\n")
            events = graph.stream(current_state, config=config, stream_mode="values")
            for event in events:
                event["messages"][-1].pretty_print()
                if event.get("query"):
                    print("Query: ", event["query"])
    except KeyboardInterrupt:
        print("\nGoodbye!")
        pass

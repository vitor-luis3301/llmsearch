import os
from dotenv import load_dotenv
load_dotenv()

#langchain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

model = ChatOllama(model=os.getenv("DEFAULT_MODEL"))

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model

def call_model(state: MessagesState):   
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

from search import search

def chat():
    config = {"configurable": {"thread_id": "testingtrack"}}

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        if query.lower() == "search":
            search(model, memory)
        else:
            messages = [HumanMessage(query)]
            ai_msg = app.invoke({"messages": messages}, config)
            
            ai_msg["messages"][-1].pretty_print()
import os
from dotenv import load_dotenv
load_dotenv()

#langchain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

#local libraries
from custom_tools.calculator import calc_tool

model = ChatOllama(model=os.getenv("DEFAULT_MODEL"))
llm = model.bind_tools([calc_tool], tool_choice="any")

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model

def call_model(state: MessagesState):   
    response = llm.invoke(state["messages"])
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
            search(model, query, memory)
        else:
            messages = [HumanMessage(query)]
            ai_msg = app.invoke({"messages": messages}, config)
            
            if ai_msg["messages"][1].tool_calls:
                messages.append(ai_msg["messages"][1])

                for tool_call in ai_msg["messages"][1].tool_calls:
                    selected_tool = {"calculator": calc_tool}[tool_call["name"].lower()]
                    tool_msg = selected_tool.invoke(tool_call)
                    messages.append(tool_msg)
                    
                output = app.invoke({"messages": messages}, config)
                output["messages"][-1].pretty_print()
            else:
                ai_msg["messages"][-1].pretty_print()
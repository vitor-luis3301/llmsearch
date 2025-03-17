import json

# Import relevant functionality
from langchain_community.tools import DuckDuckGoSearchResults, Tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langgraph.prebuilt import create_react_agent

def search(model, memory):
    ddg_search = DuckDuckGoSearchResults(output_format="list")

    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=ddg_search.run,
            description=(
                "A wrapper around the Duck Duck Go search engine. "
                "Useful for when user asks questions about anything on the internet. "
                "Use by default. "
            )
        )
    ]


    agent = create_react_agent(model, tools, checkpointer=memory)

    # Use the agent
    config = {"configurable": {"thread_id": "testingtrack"}}
    links = []
    
    while True:
        query = input("Search: ")
        
        if query.lower() == "exit":
            break
        else:
            for chunk, metadata in agent.stream({"messages": [HumanMessage(content=query)]}, config, stream_mode="messages"):
                #print(chunk)
                if isinstance(chunk, AIMessage) and chunk.content not in ["", '', ' ', " "]:
                    print(chunk.content)
                    print("References: ", end="")
                    for i in links:
                        print(i, " ", end="")
                if isinstance(chunk, ToolMessage):
                    if chunk.name == "DuckDuckGo Search":
                        search_res = json.loads(chunk.content)
                        for i in search_res:
                            links.append(i["link"])
                    else:
                        links.append(chunk.name)
                print("\n")


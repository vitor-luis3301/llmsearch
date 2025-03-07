from langchain_community.tools import WikipediaQueryRun, Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.prebuilt import create_react_agent

def chat(model, query, memory):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    tools = [
        Tool(
            name="Wikipedia Search",
            func=wikipedia.run,
            description=(
                "A wrapper around Wikipedia. "
                "Useful for when you need to answer general questions about "
                "people, places, companies, facts, historical events, or other subjects. "
                "Input should be a search query."
            ),
        )
    ]
    
    agent = create_react_agent(model, tools, checkpointer=memory)

    # Use the agent
    config = {"configurable": {"thread_id": "testingtrack"}}
    links = []

    for chunk, metadata in agent.stream({"messages": [HumanMessage(content=query)]}, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage) and chunk.content not in ["", '', ' ', " "]:
            print(chunk.content)
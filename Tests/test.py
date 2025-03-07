import os, sys
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits.load_tools import load_tools, Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

from custom_tools.calculator import calc_tool

ollama_llm = ChatOllama(model=os.getenv("DEFAULT_MODEL"), temperature=0)

react_template = """Answer the following question as best as you can. You have access to the following tools:

{tools}

Use the following format:

Question : The input question you must answer to
Thought : you should always think about what to do, do not use any tool if it is not needed. 
Action : The action to take, should be one of [{tool_names}]
Action Input : the input to the action
Observation : the result of the action
.... (this thought/Action/Action input/Observation can repeat N times)
Thought: I know the final answer
Final Answer: The final answer to the user's question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(template = react_template, input_variables=["tools", "tool_names", "input", "agent_scratchpad"])


search = DuckDuckGoSearchResults()
search_tool = Tool(
    name = "DuckDuck search tool",
    description=(
        "A web search engine. Use this to seach engine for a general queries. "
        "Do not use it if user does not ask to 'search' for something."
    ),
    func=search.run,
)

# Prepare tools
tools = load_tools([], llm=ollama_llm, allow_dangerous_tools=True)
tools.append(calc_tool)
tools.append(search_tool)

agent = create_react_agent(ollama_llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

while True:
    queue = input("You: ")
    
    if queue.lower() == "exit":
        break
    else:
        output = agent_executor.invoke(
            {
                "input": queue
            }
        )

    print(output["output"])
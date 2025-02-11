import os
from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_csv_agent

load_dotenv()

"""
The main execution function
"""

def main():
    print("start...")
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Now create the tool list
    tool = [PythonREPLTool()]
    # create the agent
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                       model="gpt-4-turbo", temperature=0.0),
                        tools=tool)
    # define the executor
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tool, verbose=True)

    # Now we will define a CSV agent executor (it is built on pandas dataframe agent)
    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, model="gpt-4-turbo"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    ### Now we will build the Grand Router Agent ###################################

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    # First create the tools for the Router Agent
    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""Useful when you need to transform natural language to python code and execute the python code. It returns the results of the code execution
            in natural language. DOES NOT ACCEPT CODE AS INPUT"""
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""Useful when you need to answer question over episode_info.csv file. Takes an input the entire question and returns the answer after running
            pandas calculations"""
        )
    ]
    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(
        grand_agent_executor.invoke(
            {"input": "Which Season has the most episodes?"}
        )
    )

    print(
        grand_agent_executor.invoke(
            {"input": "Generate and save in the directory qrcodes, 5 QRCode that point to `www.adobe.com`"}
        )
    )


if __name__ == '__main__':
    main()





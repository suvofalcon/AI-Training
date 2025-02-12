from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

@tool
def multiply(x:float, y:float) -> float:
    """Multiply x times y"""
    return x * y

if __name__ == '__main__':
    print("Tool Calling")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assistant!!"),
        ("human","{input}"),
        ("placeholder","{agent_scratchpad}")
    ])

    tools = [TavilySearchResults(), multiply]

    llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022",
                        temperature=0.0,
                        timeout=None,
                        max_tokens_to_sample=1024,
                        max_retries=2,
                        stop=None)

    # Create the agent and executor
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    result = agent_executor.invoke(
        {
            "input":"What is the weather in dubai right now? compare it with San Fransisco, output should be in Celsius.. "
        }
    )

    print(result['output'][0]['text'])
import os
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.serpapi_tools import SerpApiTools

def agent_search():
    web_agent = Agent(
        name='Web Search Agent',
        role="Search the web for information",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[SerpApiTools(api_key=os.getenv("SERPAPI_API_KEY"))],
        instructions=["Always Include Sources"],
        show_tool_calls=True,
        markdown=True,
    )
    return web_agent

if __name__ == "__main__":
    agent = agent_search()
    agent.print_response("What's happening in France?", stream=True)
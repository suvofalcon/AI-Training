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

def image_search_agent():
    web_agent = Agent(
        name='Image Search Agent',
        role="Search the web for information related to images",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[SerpApiTools(api_key=os.getenv("SERPAPI_API_KEY"))],
        show_tool_calls=True,
        markdown=True,
    )
    return web_agent

if __name__ == "__main__":
    agent = agent_search()
    img_agent = image_search_agent()
    agent.print_response("What's happening in France in 2025?", stream=True)
    img_agent.print_response(
        "Tell me about this image and give me the latest news about it.",
        images=["https://upload.wikimedia.org/wikipedia/commons/b/bf/Krakow_-_Kosciol_Mariacki.jpg"],
        stream=True,
    )
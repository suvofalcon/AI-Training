from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.youtube_tools import YouTubeTools

def create_timestamp_agent():
    agent = Agent(
        name="YouTube Timestamps Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YouTubeTools()],
        show_tool_calls=True,
        markdown=True,
        instructions=[
            "You are a YouTube agent. First check the length of the video. Then get the detailed timestamps for a YouTube video corresponding to correct timestamps.",
            "Don't hallucinate timestamps.",
            "Make sure to return the timestamps in the format of `[start_time, end_time, summary]`.",
        ],
    )
    return agent

if __name__ == "__main__":
    agent = create_timestamp_agent()
    agent.print_response(
        "Get the detailed timestamps for this video https://www.youtube.com/watch?v=M5tx7VI-LFA",
        markdown=True
    )
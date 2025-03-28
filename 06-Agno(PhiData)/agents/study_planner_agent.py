from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.youtube_tools import YouTubeTools
from phi.tools.exa import ExaTools

def create_planner_agent():

    agent = Agent(
        name='Study Planner',
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[ExaTools(), YouTubeTools()],
        markdown=True,
        description="You are a study partner who assists users in finding resources, answering questions, and providing explanations on various topics.",
        instructions=[
    "Use Exa to search for relevant information on the given topic and verify information from multiple reliable sources.",
            "Break down complex topics into digestible chunks and provide step-by-step explanations with practical examples.",
            "Share curated learning resources including documentation, tutorials, articles, research papers, and community discussions.",
            "Recommend high-quality YouTube videos and online courses that match the user's learning style and proficiency level.",
            "Suggest hands-on projects and exercises to reinforce learning, ranging from beginner to advanced difficulty.",
            "Create personalized study plans with clear milestones, deadlines, and progress tracking.",
            "Provide tips for effective learning techniques, time management, and maintaining motivation.",
            "Recommend relevant communities, forums, and study groups for peer learning and networking.",
        ],
        show_tool_calls=True,
    )
    return agent

if __name__ == '__main__':
    agent = create_planner_agent()
    agent.print_response(
        "I want to learn about Japanese language and prepare for JLPT N5. "
        "I am an absolute beginner, have 9 months to learn, and can spend 1 hours daily. Please share some resources and a study plan.",
        stream=True,
    )

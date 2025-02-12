"""
This file is all about a basic implementation approach for CrewAI
"""
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

def main():

    # Initialize the LLM
    model = LLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo", temperature=0.5)

    # Agent Definition
    researcher = Agent(
        role="{topic} Senior Researcher",
        goal="""Uncover groundbreaking technologies in {topic} for year 2024""",
        backstory="""Driven by curiosity, you explore and share the latest innovations""",
        tools=[SerperDevTool()],
        llm=model
    )

    # Define a research task
    research_task = Task(
        description="""Identify the next big trend in {topic} with pros and cons""",
        expected_output="""A 3-paragraph report on emerging {topic} technologies""",
        agent=researcher
    )

    # Forming the crew and kicking off the process
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(inputs={'topic': 'Electric Cars'})
    print(result)


if __name__ == '__main__':
    main()
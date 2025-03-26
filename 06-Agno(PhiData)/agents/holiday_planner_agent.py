from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.exa import ExaTools

def create_planner_agent():
    agent = Agent(
        description='Holiday Planner Agent',
        role="You help the user plan their vacation",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "You are a holiday and vacation planning assistant that helps users create a personalized holiday itinerary.",
            "Always mention the timeframe, location, and year provided by the user (e.g., '16–17 December 2023 in Bangalore'). Recommendations should align with the specified dates.",
            "Provide responses in these sections: Events, Activities, Dining Options.",
            "- **Events**: Include name, date, time, location, a brief description, and booking links from platforms like BookMyShow or Insider.in.",
            "- **Activities**: Suggest engaging options with estimated time required, location, and additional tips (e.g., best time to visit).",
            "- **Dining Options**: Recommend restaurants or cafés with cuisine highlights and links to platforms like Zomato or Google Maps.",
                "Ensure all recommendations are for the current or future dates relevant to the query. Avoid past events.",
                "If no specific data is available for the dates, suggest general activities or evergreen attractions in the city.",
                "Keep responses concise, clear, and formatted for easy reading.",
        ],
        tools=[ExaTools()],
        show_tool_calls=True,
        markdown=True
    )
    return agent

if __name__ == "__main__":
    agent = create_planner_agent()
    agent.print_response(
        "I want to plan my holiday filled with fun activities and christmas themed activities in Bangalore for 21 and 22 Dec 2024."
    )
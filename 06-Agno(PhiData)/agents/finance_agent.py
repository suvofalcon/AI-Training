from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.googlesearch import GoogleSearch
from phi.tools.yfinance import YFinanceTools

def create_sentiment_agent():
    sentiment_agent = Agent(
        name="Sentiment Agent",
        role="Search and interpret news articles.",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[GoogleSearch()],
        instructions=[
            "Find relevant news articles for each company and analyze the sentiment.",
            "Provide sentiment scores from 1 (negative) to 5 (positive) with reasoning and sources."
            "Cite your sources. Be specific and provide links."
        ],
        show_tool_calls=True,
        markdown=True
    )
    return sentiment_agent

def create_finance_agent():
    finance_agent = Agent(
        name='Finance Agent',
        role="Get Financial Data and interpret trends.",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
        instructions=[
            "Retrieve stock prices, analyst recommendations, and key financial data.",
            "Focus on trends and present the data in tables with key insights."
        ],
        show_tool_calls=True,
        markdown=True
    )
    return finance_agent

def create_analyst_agent():
    analyst_agent = Agent(
        name='Analyst Agent',
        role="Ensure thoroughness and draw conclusions.",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Check outputs for accuracy and completeness.",
            "Synthesize data to provide a final sentiment score (1-5) with justification."
        ],
        show_tool_calls=True,
        markdown=True
    )
    return analyst_agent

def create_agent_team():
    team = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        team=[create_sentiment_agent(), create_finance_agent(), create_analyst_agent()],
        instructions=[
            "Combine the expertise of all agents to provide a cohesive, well-supported response.",
            "Always include references and dates for all data points and sources.",
            "Present all data in structured tables for clarity.",
            "Explain the methodology used to arrive at the sentiment scores."
        ],
        show_tool_calls=True,
        markdown=True
    )
    return team

if __name__ == '__main__':
    agent_team = create_agent_team()

    agent_team.print_response(
        "Analyze the sentiment for the following companies during the week of December 2nd-6th, 2024: NVDA, MSFT. \n\n"
        "1. **Sentiment Analysis**: Search for relevant news articles and interpret th–e sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n"
        "2. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Highlight key trends or events, and present the data in tables.\n\n"
        "3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings.\n\n"
        "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates.",
        stream=True
    )



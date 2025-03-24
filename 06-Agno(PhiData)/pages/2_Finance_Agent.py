import streamlit as st
from agno.run.response import RunResponse

from agents import finance_agent

st.set_page_config(page_title="Finance Agent")

st.title("Stock Sentiment Agent using PhiData :chart:")
st.markdown("---")

st.text("Give your Stock Symbol")
stock_ticker = st.text_input("Stock Ticker")

if st.button("Recommendation"):
    team = finance_agent.create_agent_team()
    response: RunResponse = team.run(f"Analyze the sentiment for the following companies during the week of December 2nd-6th, 2024: {stock_ticker}. \n\n"
        "1. **Sentiment Analysis**: Search for relevant news articles and interpret th–e sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n"
        "2. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Highlight key trends or events, and present the data in tables.\n\n"
        "3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-5) for each company. Justify the scores and provide a summary of the most important findings.\n\n"
        "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates.")
    st.write(response.content)
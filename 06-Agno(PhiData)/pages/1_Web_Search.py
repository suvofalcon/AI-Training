import streamlit as st

import time
import numpy as np
from agno.run.response import RunResponse

from agents import web_search_agent

st.set_page_config(page_title="Web Search")

st.title("Web Search using Agnos Agent")
st.markdown("---")

search_text = st.text_area("Give your search here")

if st.button("Search"):
    agent = web_search_agent.agent_search()
    response: RunResponse = agent.run(search_text)
    st.write(response.content)


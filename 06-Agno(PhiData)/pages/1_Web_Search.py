import streamlit as st

import time
import numpy as np
from agents import web_search_agent

st.set_page_config(page_title="Web Search")

st.title("Web Search using Agnos Agent")
st.markdown("---")

st.text_area("Give your search here")

if st.button("Search"):
    agent = web_search_agent.agent_search()

    st.write(agent.print_response("Whats happening in France?", stream=True))

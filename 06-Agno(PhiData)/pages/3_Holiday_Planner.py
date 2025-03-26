import streamlit as st
from datetime import datetime
from agno.run.response import RunResponse
from agents import holiday_planner_agent

st.set_page_config(page_title="Holiday Activities Planner Agent")

st.title("Holiday Activities Planning Agent :camping:")
st.markdown("---")

start_date = st.date_input("Holiday Start Date", value="today", min_value="today", max_value=None)
end_date = st.date_input("Holiday End Date", value="today", min_value="today", max_value=None)
destination_places = st.text_area("Give the Destination Places")

if st.button("Plan Holiday"):
    planner_agent = holiday_planner_agent.create_planner_agent()
    start_date = start_date.strftime('%d-%m-%Y')
    end_date = end_date.strftime('%d-%m-%Y')

    response: RunResponse = planner_agent.run(f"I want to plan my vacation filled with fun activities for the destination places {destination_places}, from {start_date} to {end_date}")
    st.write(response.content)

    # st.write(start_date)
    # st.write(end_date)
    # st.write(destination_places)
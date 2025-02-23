#!/usr/bin/env python
import sys
import os
import warnings
from dotenv import load_dotenv

from datetime import datetime
import agentops
from crew import CodeExecution

load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
DATA_PATH = os.path.join(os.getenv("DATASETS_PATH"),'County_Health_Rankings.csv')

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'file_path': DATA_PATH
    }
    
    try:
        result = CodeExecution().crew().kickoff(inputs=inputs)
        print(result)

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":

    session = agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))
    run()
    session.end_session()
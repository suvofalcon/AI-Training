#!/usr/bin/env python
import sys
import os
import warnings
from datetime import datetime
import agentops
from dotenv import load_dotenv
from crew import PdfRag

load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    user_input = input("Enter your question: ")
    inputs = {
        'input': user_input
    }

    try:
        result = PdfRag().crew().kickoff(inputs=inputs)
        print(result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == '__main__':
    session = agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))
    run()
    session.end_session()
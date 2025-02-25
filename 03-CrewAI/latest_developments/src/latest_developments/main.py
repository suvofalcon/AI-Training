#!/usr/bin/env python
import os
import warnings
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Setup Langtrace
from langtrace_python_sdk import langtrace
langtrace.init(api_key=os.environ["LANGTRACE_API_KEY"])

from latest_developments.src.latest_developments.crew import LatestDevelopments

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'Electric Vehicles in India',
        'current_year': str(datetime.now().year)
    }
    
    try:
        LatestDevelopments().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


# def train():
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {
#         "topic": "AI LLMs"
#     }
#     try:
#         LatestDevelopments().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
#
#     except Exception as e:
#         raise Exception(f"An error occurred while training the crew: {e}")
#
# def replay():
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         LatestDevelopments().crew().replay(task_id=sys.argv[1])
#
#     except Exception as e:
#         raise Exception(f"An error occurred while replaying the crew: {e}")
#
# def test():
#     """
#     Test the crew execution and returns the results.
#     """
#     inputs = {
#         "topic": "AI LLMs"
#     }
#     try:
#         LatestDevelopments().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
#
#     except Exception as e:
#         raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()
#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
from datetime import datetime
from crew import  PdfRag1


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

load_dotenv()

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    user_input = input("Which section of the report would you like to generate a work order for?\n: ")
    inputs = {
        'customer_question': user_input
    }
    try:
        result = PdfRag1().crew().kickoff(inputs=inputs)
        print(result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == '__main__':
    run()


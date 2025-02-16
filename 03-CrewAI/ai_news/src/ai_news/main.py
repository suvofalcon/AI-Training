#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import AiNews

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
        'topic': 'Satyajit Ray',
        'current_year': str(datetime.now().year)
    }

    inputs_array = [
        {
            'topic': 'electric vehicles',
            'date': datetime.now().strftime('%Y-%m-%d_%H-%M-%S_1')
        },
        {
            'topic': 'nvidia RTX',
            'date': datetime.now().strftime('%Y-%m-%d_%H-%M-%S_2')
        },
        {
          'topic': 'Linux Kernel',
          'date': datetime.now().strftime('%Y-%m-%d_%H-%M-%S_3')
        }
    ]

    try:
        #AiNews().crew().kickoff(inputs=inputs)
        AiNews().crew().kickoff_for_each(inputs=inputs_array)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == '__main__':
    run()
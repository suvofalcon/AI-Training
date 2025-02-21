#!/usr/bin/env python
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks
from crewai.flow import Flow, listen, start
from crews.meeting_minutes_crew.meeting_minutes_crew import MeetingMinutesCrew

load_dotenv()

class MeetingMinutesState(BaseModel):
    transcript: str = ""
    meeting_minutes: str = ""

class MeetingMinutesFlow(Flow[MeetingMinutesState]):

    @start()
    def transcribe_meeting(self):
        print("Generating Transcription")

        # Load the audio file
        audio = AudioSegment.from_file("EarningsCall.wav", format="wav")

        # Define chunk length in milliseconds (e.g 1 minute = 60,000 ms)
        chunk_length = 60000
        chunks = make_chunks(audio, chunk_length)

        # Transcribe each chunk
        full_transcription = ""
        for i, chunk in enumerate(chunks):
            print(f"Transcribing Chunk {i+1} / {len(chunks)}")
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            with open(chunk_path, 'rb') as audio_file:
                transcription = OpenAI().audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                full_transcription += transcription.text + " "

        self.state.transcript = full_transcription
        print(f"Transcription Completed - {self.state.transcript}")


    @listen(transcribe_meeting)
    def generate_meeting_minutes(self):
        print("Generating Meeting Minutes")

        crew = MeetingMinutesCrew()
        inputs = {
            "transcript": self.state.transcript
        }

        meeting_minutes = crew.crew().kickoff(inputs=inputs)
        self.state.meeting_minutes = meeting_minutes
        print(f"Meeting Minutes Completed - {self.state.meeting_minutes}")


    @listen(generate_meeting_minutes)
    def create_draft_meeting_minutes(self):
        print("Creating Draft Meeting Minutes")


def kickoff():
    meeting_minutes_flow = MeetingMinutesFlow()
    meeting_minutes_flow.kickoff()


def plot():
    meeting_minutes_flow = MeetingMinutesFlow()
    meeting_minutes_flow.plot()


if __name__ == "__main__":
    kickoff()

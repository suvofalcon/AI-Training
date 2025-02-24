import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

pdf_file_path = os.path.join(os.getenv("DATASETS_PATH"), "example_home_inspection.pdf")
pdf_search_tool = PDFSearchTool(pdf=pdf_file_path)

@CrewBase
class PdfRag1():
	"""PdfRag1 crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def research_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['research_agent'],
			tools=[pdf_search_tool],
			verbose=True
		)

	@agent
	def professional_writer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['professional_writer_agent'],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def answer_customer_question_task(self) -> Task:
		return Task(
			config=self.tasks_config['answer_customer_question_task'],
			tools=[pdf_search_tool]
		)

	@task
	def write_email_task(self) -> Task:
		return Task(
			config=self.tasks_config['write_email_task']
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the PdfRag1 crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

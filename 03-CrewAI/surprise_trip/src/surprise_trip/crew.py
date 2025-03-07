from typing import Optional, List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class Activity(BaseModel):
	name: str = Field(..., description='Name of the activity')
	location: str = Field(..., description='Location of the activity')
	description: str = Field(..., description='Description of the activity')
	date: str = Field(..., description='Date of the activity')
	cuisine: str = Field(..., description='Cuisine of the restaurant')
	why_its_suitable: str = Field(..., description='Why it is suitable for the traveller')
	reviews: Optional[str] = Field(None, description='List of reviews')
	rating: Optional[float] = Field(None, description='Rating of the activity')

class DayPlan(BaseModel):
	date: str = Field(..., description='Date of the day')
	activities: List[Activity] = Field(..., description='List of activities')
	restaurants: List[str] = Field(..., description='List of restaurants')
	flight: Optional[str] = Field(None, description='Flight information')

class Itinerary(BaseModel):
	name: str = Field(..., description='Name of the itinerary, something funny name suggested')
	day_plans: List[DayPlan] = Field(..., description='List of day_plans')
	hotel: Optional[str] = Field(None, description='Hotel information')

@CrewBase
class SurpriseTripCrew():
	"""SurpriseTrip Crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def personalized_activity_planner(self) -> Agent:
		return Agent(
			config=self.agents_config['personalized_activity_planner'],
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			allow_delegation=False,
			verbose=True
		)

	@agent
	def restaurant_scout(self) -> Agent:
		return Agent(
			config=self.agents_config['restaurant_scout'],
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			allow_delegation=False,
			verbose=True
		)

	@agent
	def itinerary_compiler(self) -> Agent:
		return Agent(
			config=self.agents_config['itinerary_compiler'],
			tools=[SerperDevTool()],
			allow_delegation=False,
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task

	@task
	def personalized_activity_planning_task(self) -> Task:
		return Task(
			config=self.tasks_config['personalized_activity_planning_task'],
			agent=self.personalized_activity_planner(),
		)

	@task
	def restaurant_scenic_location_scout_task(self) -> Task:
		return Task(
			config=self.tasks_config['restaurant_scenic_location_scout_task'],
			agent=self.restaurant_scout(),
		)

	@task
	def itinerary_compilation_task(self) -> Task:
		return Task(
			config=self.tasks_config['itinerary_compilation_task'],
			agent=self.itinerary_compiler(),
			output_json=Itinerary
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the SurpriseTrip crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

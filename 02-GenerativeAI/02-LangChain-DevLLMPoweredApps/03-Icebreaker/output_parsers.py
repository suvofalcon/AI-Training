from typing import List, Dict, Any

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Lets create the pydantic object - this will help us to serialise the output later
# It defines the schema of the output from the LLM

class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}


# Now lets create the pydantic output parser - this will take our earlier created pydantic object

summary_parser = PydanticOutputParser(pydantic_object=Summary)



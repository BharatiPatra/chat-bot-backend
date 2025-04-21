from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from typing import List, Optional
from utils.constants import MODEL_ID


# --------- Schema ---------
class Project(BaseModel):
    name: Optional[str] = Field(description="Name of the project")
    description: Optional[str] = Field(description="Description of the project")

class Education(BaseModel):
    degree: Optional[str] = Field(description="Degree obtained")
    institution: Optional[str] = Field(description="Name of the institution")
    year: Optional[int] = Field(description="Year of graduation")

class Contact(BaseModel):
    email: Optional[str] = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number")

class ResumeInfo(BaseModel):
    name: str = Field(description="Name of the candidate")
    age: Optional[int] = Field(description="Age of the candidate")
    experience_years: Optional[float] = Field(description="Years of experience")
    skills: List[str] = Field(description="List of skills")
    projects: List[Project] = Field(description="List of projects")
    education: List[Education] = Field(description="List of education details")
    contact: Contact = Field(description="Contact information")
    location: Optional[str] = Field(description="Location of the candidate")

# --------- Agent ---------
load_dotenv()
model_id = MODEL_ID

class ProfileReader():
    def __init__(self, file_path: str):
        self.llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=ResumeInfo)
        self.prompt = PromptTemplate(
            template="""
            You are an AI assistant. Extract structured resume data from the user details below.

            {format_instructions}

            Here is the resume content:
            {query}
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.prompt_and_model = self.prompt | self.llm | self.parser
        self.file_path = file_path

    def extract_text(self) -> str:
        # Replace with actual text extraction
        profile_reader = ProfileReader(self.file_path)
        text = profile_reader.extract_text()
        return text

    def parse_resume(self, text: str) -> dict:
        out = self.prompt_and_model.invoke({"query": text})
        json_string = out.model_dump_json(indent=2)
        print(json_string)
        return out

# --------- Run Test ---------
if __name__ == "__main__":
    profile_reader = ProfileReader(file_path="resume.pdf")
    text = profile_reader.extract_text()
    parsed_data = profile_reader.parse_resume(text)
    print(parsed_data)

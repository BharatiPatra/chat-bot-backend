from pydantic import BaseModel, Field
from typing import List, Optional

class Project(BaseModel):
    name: str = Field(description="Name of the project")
    description: str = Field(description="Description of the project")

class Education(BaseModel):
    degree: str = Field(description="Degree obtained")
    institution: str = Field(description="Name of the institution")
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

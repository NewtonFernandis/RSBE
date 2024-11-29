import os
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Form,WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
import PyPDF2
import pypdf
import docx
import textract
import uuid
from resumeRanker import ResumeRanker
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from pydantic import BaseModel
import asyncio



class ApplicantProfile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    designation: Optional[str] = None
    skills: List[str] = []
    experience_level: Optional[str] = None
    preferred_locations: List[str] = []

class JobListing(BaseModel):
    title: str
    company: str
    location: str
    link: str

class InteractiveJobChatbot:
    def __init__(self):
        self.conversation_stages = [
            "welcome",
            "get_name",
            "get_email",
            "get_designation",
            "get_experience_level",
            "get_skills",
            "get_locations",
            "job_search"
        ]
        self.conversation_states: Dict[str, Dict] = {}

    def scrape_linkedin_jobs(self, keyword: str, location: str = 'United States') -> List[JobListing]:
        """Simplified job scraping method"""
        try:
            encoded_keyword = quote(keyword)
            encoded_location = quote(location)
            url = f"https://www.linkedin.com/jobs/search?keywords={encoded_keyword}&location={encoded_location}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            jobs = []
            for job in soup.find_all('div', class_='base-card')[:5]:  # Limit to 5 jobs
                try:
                    job_listing = JobListing(
                        title=job.find('h3', class_='base-search-card__title').text.strip(),
                        company=job.find('h4', class_='base-search-card__subtitle').text.strip(),
                        location=job.find('span', class_='job-search-card__location').text.strip(),
                        link=job.find('a', class_='base-card__full-link')['href']
                    )
                    jobs.append(job_listing)
                except Exception:
                    continue
            
            return jobs
        except Exception as e:
            print(f"Job scraping error: {e}")
            return []

    async def handle_conversation(self, websocket: WebSocket, session_id: str):
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = {
                "profile": ApplicantProfile(),
                "current_stage": 0
            }

        state = self.conversation_states[session_id]
        current_stage = self.conversation_stages[state['current_stage']]
        profile = state['profile']

        responses = {
            "welcome": "👋 Welcome to JobMate! I'll help you find your perfect job. What's your name?",
            "get_name": f"Nice to meet you! What's your email address?",
            "get_email": "What job designation are you looking for? (e.g., Software Engineer, Data Scientist)",
            "get_designation": "What's your experience level? (Entry, Mid, Senior)",
            "get_experience_level": "What are your key skills? (Separate by commas)",
            "get_skills": "In which locations are you interested in working?",
            "get_locations": "Great! Let me find some matching jobs for you...",
            "job_search": self.generate_job_recommendations(profile)
        }

        await websocket.send_text(responses[current_stage])

    def process_user_input(self, session_id: str, user_input: str):
        state = self.conversation_states[session_id]
        current_stage = self.conversation_stages[state['current_stage']]
        profile = state['profile']

        input_processors = {
            "get_name": lambda: setattr(profile, 'name', user_input),
            "get_email": lambda: setattr(profile, 'email', user_input),
            "get_designation": lambda: setattr(profile, 'designation', user_input),
            "get_experience_level": lambda: setattr(profile, 'experience_level', user_input),
            "get_skills": lambda: setattr(profile, 'skills', [skill.strip() for skill in user_input.split(',')]),
            "get_locations": lambda: setattr(profile, 'preferred_locations', [loc.strip() for loc in user_input.split(',')])
        }

        if current_stage in input_processors:
            input_processors[current_stage]()
            state['current_stage'] += 1

        return state

    def generate_job_recommendations(self, profile: ApplicantProfile) -> str:
        # Use the first preferred location or default to United States
        location = profile.preferred_locations[0] if profile.preferred_locations else 'United States'
        
        # Scrape jobs based on designation
        jobs = self.scrape_linkedin_jobs(profile.designation or 'Software Engineer', location)
        
        if not jobs:
            return "Sorry, no jobs match your current profile. Would you like to try a different search?"

        recommendation = "🌟 Job Recommendations:\n"
        for idx, job in enumerate(jobs, 1):
            recommendation += f"{idx}. {job.title} at {job.company}, {job.location}\n   Link: {job.link}\n"
        
        recommendation += "\nWould you like to apply to any of these jobs or refine your search?"
        return recommendation

chatbot = InteractiveJobChatbot()

# Create FastAPI application with enhanced configurations
app = FastAPI(
    title="AI-Powered Resume Ranking Service",
    description="Advanced backend for ranking resumes against job descriptions",
    version="1.0.0"
)

# Add CORS middleware for cross-origin support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global resume ranker instance
resume_ranker = ResumeRanker()

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/rank-resumes/")
async def rank_resumes(
    job_description: str = Form(...),
    resumes: List[UploadFile] = File(...)
):
    """
    API endpoint for ranking resumes against a job description.
    
    Args:
        job_description (str): Detailed job description text
        resumes (List[UploadFile]): List of resume files to be ranked
    
    Returns:
        JSONResponse with ranked resumes
    """
    try:
        # Save uploaded files
        resume_paths = []
        for resume in resumes:
            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}_{resume.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                buffer.write(await resume.read())
            
            resume_paths.append(file_path)
        
        # Rank resumes
        ranked_resumes = resume_ranker.rank_resumes(job_description, resume_paths)
        
        return JSONResponse(content={
            "message": "Resumes ranked successfully",
            "ranked_resumes": ranked_resumes
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        await chatbot.handle_conversation(websocket, session_id)
        
        while True:
            user_input = await websocket.receive_text()
            chatbot.process_user_input(session_id, user_input)
            await chatbot.handle_conversation(websocket, session_id)
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

# Configuration for running the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
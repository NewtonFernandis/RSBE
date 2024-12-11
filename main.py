import os
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
import uuid
import urllib.parse
from urllib.request import Request, urlopen
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from chatManager import ChatManager
from jobScraper import JobScraper
 
# Create FastAPI application with enhanced configurations
app = FastAPI()

# Add CORS middleware for cross-origin support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatbotState:
    def __init__(self):
        self.scraper = JobScraper()
        self.chat_manager = ChatManager()

chat_state = ChatbotState()

# Global resume ranker instance
resume_ranker = ResumeRanker()

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Process chat message
            response = chat_state.chat_manager.process_message(data)
            
            # Check if response is a search command
            if response.startswith("/search"):
                _, job_title, job_location = response.split("|")
                
                # Send initial confirmation
                await websocket.send_text(f"Searching for {job_title} jobs in {job_location}...")
                
                # Scrape jobs
                jobs = await chat_state.scraper.scrape_indeed(job_title, job_location)
                
                # Set current jobs in chat manager
                chat_state.chat_manager.set_current_jobs(jobs)
                
                # Generate summary
                summary = f"I found {len(jobs)} job postings. Would you like more details?"
                await websocket.send_text(summary)
            else:
                # Send regular chat response
                await websocket.send_text(response)
    
    except WebSocketDisconnect:
        print("WebSocket connection closed")

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
            unique_filename = f"{uuid.uuid4()}?{resume.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                buffer.write(await resume.read())
            
            resume_paths.append(file_path)
        
        # Rank resumes
        ranked_resumes = resume_ranker.rank_resumes(job_description, resume_paths)

        for file_path in resume_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return JSONResponse(content={
            "message": "Resumes ranked successfully",
            "ranked_resumes": ranked_resumes
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
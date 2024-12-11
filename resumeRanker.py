from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import PyPDF2
import pypdf
import textract
class ResumeRanker:
    """
    Advanced resume ranking service with modular design and extensible architecture.
    Supports multiple file formats and uses advanced text processing techniques.
    """
    
    def __init__(self):
        """
        Initialize the resume ranking system with core vectorization components.
        
        Design allows for future enhancement of similarity computation methods.
        """
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from various file formats.
        
        Supports:
        - PDF documents
        - Microsoft Word documents (.docx)
        - Plain text files
        - Other formats via textract
        
        Args:
            file_path (str): Path to the resume file
        
        Returns:
            str: Extracted and cleaned text content
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ' '.join([page.extract_text() for page in reader.pages])
            elif file_extension == '.docx':
                doc = docx.Document(file_path)
                text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            else:
                # Fallback to textract for other file types
                text = textract.process(file_path).decode('utf-8')
            
            return self._clean_text(text)
        
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Advanced text cleaning and normalization.
        
        Transformations:
        - Remove special characters
        - Convert to lowercase
        - Remove extra whitespaces
        
        Args:
            text (str): Raw input text
        
        Returns:
            str: Cleaned and normalized text
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def rank_resumes(self, job_description: str, resume_files: List[str]) -> List[Dict[str, float]]:
        """
        Sophisticated resume ranking using TF-IDF and cosine similarity.
        
        Args:
            job_description (str): Detailed job description text
            resume_files (List[str]): Paths to resume files
        
        Returns:
            List[Dict[str, float]]: Ranked resumes with metadata
        """
        cleaned_jd = self._clean_text(job_description)
        
        resume_texts = [
            (file, self.extract_text_from_file(file)) 
            for file in resume_files
        ]
        
        resume_texts = [
            (file, text) for file, text in resume_texts if text
        ]
        
        texts = [cleaned_jd] + [text for _, text in resume_texts]
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        ranked_resumes = [
            {
                "filename": os.path.basename(file).split('?')[-1],
                "filepath": file,
                "similarity_score": float(score)
            }
            for (file, _), score in zip(resume_texts, similarities)
        ]
        
        return sorted(ranked_resumes, key=lambda x: x['similarity_score'], reverse=True)
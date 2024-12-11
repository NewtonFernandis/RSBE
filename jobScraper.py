import urllib.parse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import asyncio
import logging

class JobScraper:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.unique_jobs = set()
    
    async def scrape_indeed(self, job_title, job_location):
        self.logger.info(f"Starting to scrape Indeed for `{job_title}` in `{job_location}`")
        
        url = self._generate_url(job_title, job_location)
        self.logger.info(f"Scraping URL: {url}")
        
        html = self._request_jobs_from_indeed(url)
        if not html:
            self.logger.warning("No HTML content retrieved")
            return []
        
        cards = self._collect_job_cards_from_page(html)
        jobs = []
        
        for card in cards:
            try:
                job = self._extract_job_card_data(card)
                
                # Only add if job is not None and URL is unique
                if job and job.get('url'):
                    if job['url'] not in self.unique_jobs:
                        self.unique_jobs.add(job['url'])
                        jobs.append(job)
            except Exception as e:
                self.logger.error(f"Error processing job card: {e}")
        
        self.logger.info(f'Finished collecting {len(jobs):,d} job postings.')
        return jobs
    
    def _generate_url(self, job_title, job_location):
        url_template = "https://www.indeed.com/jobs?q={}&l={}"
        url = url_template.format(
            urllib.parse.quote_plus(job_title), 
            urllib.parse.quote_plus(job_location)
        )
        return url
    
    def _request_jobs_from_indeed(self, url):
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--start-maximized")
        # chrome_options.add_argument("--headless")  # Run in background
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        try:
            driver.get(url)
            driver.implicitly_wait(10)
            page_source = driver.page_source
            return page_source
        
        except Exception as e:
            self.logger.error(f"Error retrieving page: {e}")
            return None
        
        finally:
            driver.quit()
    
    def _collect_job_cards_from_page(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.find_all('div', 'job_seen_beacon')
        return cards
    
    def _extract_job_card_data(self, card):
        try:
            # Carefully extract each piece of information
            job_title_tag = card.find('h2', class_='jobTitle')
            if not job_title_tag:
                return None
            
            # Use .get_text(strip=True) to safely extract text
            job_title = job_title_tag.get_text(strip=True)
            
            # Company name
            company_tag = card.find('span', class_='companyName')
            company_name = company_tag.get_text(strip=True) if company_tag else 'Unknown Company'
            
            # Job URL
            atag = job_title_tag.find('a')
            if not atag or not atag.get('href'):
                return None
            
            job_url = 'https://www.indeed.com' + atag.get('href')
            
            # Location
            location_tag = card.find('div', class_='companyLocation')
            location = location_tag.get_text(strip=True) if location_tag else 'Location Not Specified'
            
            # Salary (optional)
            salary_tag = card.find('div', class_='metadata salary-snippet-container')
            salary = salary_tag.get_text(strip=True) if salary_tag else 'Salary Not Listed'
            
            return {
                'title': job_title,
                'company': company_name,
                'url': job_url,
                'location': location,
                'salary': salary
            }
        except Exception as e:
            self.logger.error(f"Error extracting job data: {e}")
            return None
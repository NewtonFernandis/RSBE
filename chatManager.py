import random

class ChatManager:
    def __init__(self):
        self.current_state = 'START'
        self.job_search_params = {
            'job_title': None,
            'job_location': None,
            'experience_level': None,
            'remote': None
        }
        self.current_jobs = []
    
    def process_message(self, message):
        message = message.lower().strip()
        
        # State machine for collecting job search parameters
        if self.current_state == 'START':
            return self._handle_initial_interaction(message)
        
        elif self.current_state == 'WAITING_JOB_TITLE':
            return self._collect_job_title(message)
        
        elif self.current_state == 'WAITING_JOB_LOCATION':
            return self._collect_job_location(message)
        
        elif self.current_state == 'CONFIRM_SEARCH':
            return self._confirm_search(message)
        
        elif self.current_state == 'JOB_SEARCH_COMPLETE':
            return self._handle_post_search(message)
        
        # Fallback for any unexpected states
        return "I'm a bit confused. Let's start over. What kind of job are you looking for?"
    
    def _handle_initial_interaction(self, message):
        # Initiate job search conversation
        if 'job' in message or 'search' in message or 'yes' in message:
            self.current_state = 'WAITING_JOB_TITLE'
            return "Great! What job title are you interested in? (e.g., Software Engineer, Data Scientist)"
        
        return "Hi there! I can help you find jobs. Would you like to start a job search?"
    
    def _collect_job_title(self, message):
        # Validate and store job title
        if len(message) < 2 or len(message) > 100:
            return "Please enter a valid job title. For example, 'Software Engineer' or 'Marketing Manager'."
        
        self.job_search_params['job_title'] = message
        self.current_state = 'WAITING_JOB_LOCATION'
        return f"Got it! You're looking for a {message} position. Where would you like to work? Please specify a city, state, or country."
    
    def _collect_job_location(self, message):
        # Validate and store job location
        if len(message) < 2 or len(message) > 100:
            return "Please enter a valid location. For example, 'New York', 'California', or 'United States'."
        
        self.job_search_params['job_location'] = message
        self.current_state = 'CONFIRM_SEARCH'
        
        return (f"You're searching for {self.job_search_params['job_title']} jobs in {message}. ")
                # "Would you like to add any additional filters like remote work or experience level? "
                # "Type 'yes' to add filters, or 'no' to start the search.")
    
    def _confirm_search(self, message):
        message = message.lower()
        
        # if message in ['yes', 'y']:
        #     # Prompt for additional filters
        #     self.current_state = 'ADDING_FILTERS'
        #     return ("Let's add some filters. You can specify:\n"
        #             "- Remote work preference (e.g., 'remote', 'on-site', 'hybrid')\n"
        #             "- Experience level (e.g., 'entry', 'mid', 'senior')\n"
        #             "Type 'done' when you're finished adding filters.")
        
        if message in ['yes', 'y']:
            # Trigger job search
            return self._start_job_search()
        
        return "Please respond with 'yes' or 'no'."
    
    def _start_job_search(self):
        # Reset current jobs and trigger scraping
        self.current_jobs = []
        self.current_state = 'JOB_SEARCH_COMPLETE'
        
        # Return a command to trigger actual job search in main application
        return f"/search|{self.job_search_params['job_title']}|{self.job_search_params['job_location']}"
    
    def _handle_post_search(self, message):
        # After job search is complete, provide interaction options
        if 'more' in message or 'details' in message or 'yes' in message:
            return self._provide_job_details()
        
        if 'filter' in message or 'refine' in message:
            return self._filter_jobs(message)
        
        # Reset for new search
        self.current_state = 'START'
        return ("Would you like to start a new job search?")
    
    def _provide_job_details(self):
        if not self.current_jobs:
            return "No jobs to show details for. Let's start a new search."
        
        details = "Job Details:\n\n"
        for job in self.current_jobs:
            details += f"Title: {job['title']}\n"
            details += f"Company: {job['company']}\n"
            details += f"Location: {job['location']}\n"
            details += f"URL: {job['url']}\n\n"
        
        # details += "Want more information or to apply to any of these jobs?"
        return details
    
    def _filter_jobs(self, message):
        if not self.current_jobs:
            return "No jobs to filter. Let's start a new search."
        
        # Simple filtering based on keywords
        if 'remote' in message:
            remote_jobs = [job for job in self.current_jobs if 'remote' in job['title'].lower()]
            return f"Found {len(remote_jobs)} remote jobs:\n" + '\n'.join([job['title'] for job in remote_jobs])
        
        return "I can help you filter jobs. What specific criteria are you looking for?"
    
    def set_current_jobs(self, jobs):
        # Method to set current jobs after scraping
        self.current_jobs = jobs
        self.current_state = 'JOB_SEARCH_COMPLETE'
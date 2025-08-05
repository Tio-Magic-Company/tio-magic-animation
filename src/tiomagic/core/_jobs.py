"""Job tracking and management for TioMagic
Handles async job execution, progress tracking, and result retrieval
"""
from dataclasses import asdict, dataclass
from enum import Enum
import time
import traceback
from pathlib import Path
import json
from typing import Dict, Optional

from ._utils import create_timestamp

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

@dataclass
class Job:
    """Job tracking for async operations"""

    job_id: str
    feature: str
    model: str
    provider: str
    generation: Optional[Dict] = None
    creation_time: Optional[str] = None
    last_updated: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = create_timestamp()
        if self.last_updated is None:
            self.last_updated = self.creation_time
    
    # Path to the log file for persistent storage
    # Find the repository root by looking for pyproject.toml or other root indicators
    def find_repo_root():
        current_path = Path(__file__).resolve()
        while current_path.parent != current_path:  # Stop at filesystem root
            if (current_path / "pyproject.toml").exists():
                return current_path
            current_path = current_path.parent
        # Fallback: use current working directory if no pyproject.toml found
        return Path.cwd()

    repo_root = find_repo_root()
    LOG_PATH = repo_root / 'generation_log.json'

    # Ensure the log file exists and is initialized as '{"jobs": []}'
    if not LOG_PATH.exists():
        LOG_PATH.write_text('{"jobs": []}')

    @classmethod
    def setup_log(cls):
         """Ensure log file exists"""
         if not cls.LOG_PATH.exists():
              print("log file does not exist, creating log file")
              with open(cls.LOG_PATH, "w") as f:
                   f.write(json.dumps({"jobs": []}))

    @classmethod
    def load_all_jobs(cls) -> Dict[str, 'Job']:
        """Load all jobs from the log file."""
        cls.setup_log()
        try:
            with open(cls.LOG_PATH, "r") as f:
                data = json.loads(f.read())
                jobs = {}
                for job_data in data.get("jobs", []):
                    job = cls.from_dict(job_data)
                    jobs[job.job_id] = job
                return jobs
        except Exception as e:
            print(f"Error loading jobs: {e}")
            return {}

    @classmethod
    def get_job(cls, job_id: str) -> Optional['Job']:
        """Get a job by ID."""
        jobs = cls.load_all_jobs()
        return jobs.get(job_id)

    @classmethod
    def save_jobs(cls, jobs: Dict[str, 'Job']):
        """Save jobs to the log file."""
        cls.setup_log()
        with open(cls.LOG_PATH, "r") as f:
            existing_data = json.load(f)

        # Convert existing jobs to a dictionary for easy lookup
        existing_jobs_dict = {}
        for job_data in existing_data.get("jobs", []):
            existing_jobs_dict[job_data["job_id"]] = job_data

        # Update or add new jobs
        for job in jobs.values():
            job_dict = job.to_dict()
            existing_jobs_dict[job.job_id] = job_dict

        # Convert back to list
        all_jobs = list(existing_jobs_dict.values())

        with open(cls.LOG_PATH, "w") as f:
            json.dump({"jobs": all_jobs}, f, indent=2)       


    def to_dict(self) -> Dict:
        """Convert job to dictionary for storage."""
        return asdict(self)
    def update(self, **kwargs):
        # update the dictionary outside of Job class
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Job':
        """create job from dictionary"""
        return cls(**data)

    def start(self, func):
        """Start the job execution asynchronously
        Args:
            func (callable): Function to execute
        """
        self.start_time = time.time()

        try:
            # Execute the function
            result = func()

            self.generation = result
            self.save()
        except Exception as e:
            # Handle execution error
            self.error = str(e)
            self.generation['status'] = JobStatus.FAILED
            self.save()
            traceback.print_exc()

    def check_status(self, func):
        """Check the job execution asynchronously
        Args:
            func (callable): Function to execute
        """
        try:
            result = func()
            if result is not None:
                self.generation = result

        except Exception as e:
            self.error = str(e)
            self.generation['status'] = JobStatus.FAILED
            traceback.print_exc()

        self.save()
    
    def cancel_job(self, func):
        """Cancel the job execution asynchronously
        Args:
            func (callable): Function to execute
        """
        try:
            result = func()
            if result is not None:
                self.generation = result
        except Exception as e:
            self.error = str(e)
            self.generation['status'] = JobStatus.FAILED
            traceback.print_exc()

        self.save()


    def save(self):
        """Save job to log file
        """
        jobs = self.load_all_jobs()
        jobs[self.job_id] = self
        print(f"--> Job {self.job_id} saved to log file")
        self.save_jobs(jobs)







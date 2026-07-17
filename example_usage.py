from pathlib import Path
from main import ATSResumeShortlister

# Example configuration
RESUMES_FOLDER = "path/to/your/resumes/folder"  # Replace with your folder path
KEYWORDS = [
    "Python",
    "SQL",
    "Machine Learning",
    "AWS",
    "Data Analysis",
    "3+ years experience"
]
WEIGHTS = [2.0, 1.5, 2.0, 1.0, 1.0, 1.5]  # Optional weights for each keyword
MIN_THRESHOLD = 0.6  # 60% match required
USE_SEMANTIC = False  # Set to True for semantic similarity matching (slower)

if __name__ == "__main__":
    # Initialize the shortlister
    shortlister = ATSResumeShortlister(
        resumes_folder=RESUMES_FOLDER,
        keywords=KEYWORDS,
        weights=WEIGHTS,
        min_threshold=MIN_THRESHOLD,
        use_semantic=USE_SEMANTIC
    )
    
    # Process all resumes
    results = shortlister.process_all_resumes()
    
    # Generate report
    shortlister.generate_report(results)

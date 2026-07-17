import os
import re
import csv
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from fuzzywuzzy import fuzz
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from resume_parser import extract_text_from_pdf_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ats_tool.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Synonym mapping (can be extended)
SYNONYMS = {
    "ml": "machine learning",
    "js": "javascript",
    "pythonic": "python",
    "sql server": "sql",
    "aws": "amazon web services",
    "reactjs": "react",
    "nodejs": "node.js",
    "node": "node.js"
}


class ATSResumeShortlister:
    def __init__(
        self,
        resumes_folder: str,
        keywords: List[str],
        weights: List[float] = None,
        min_threshold: float = 0.6,
        fuzzy_threshold: float = 85.0,
        use_semantic: bool = False,
        semantic_threshold: float = 0.7
    ):
        self.resumes_folder = Path(resumes_folder)
        self.keywords = self._normalize_keywords(keywords)
        self.weights = weights if weights else [1.0] * len(self.keywords)
        self.min_threshold = min_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold
        self.shortlisted_folder = self.resumes_folder.parent / "shortlisted"
        self.log_folder = self.resumes_folder.parent / "logs"
        self.processed_file = self.log_folder / "processed_files.txt"
        
        # Create necessary directories
        self.shortlisted_folder.mkdir(exist_ok=True)
        self.log_folder.mkdir(exist_ok=True)
        
        # Load processed files for resumability
        self.processed_files = self._load_processed_files()
        
        # Load sentence transformer model if semantic matching is enabled
        if self.use_semantic:
            logger.info("Loading semantic similarity model...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.keyword_embeddings = self.semantic_model.encode(self.keywords, convert_to_tensor=True)
        else:
            self.semantic_model = None
            self.keyword_embeddings = None

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        normalized = []
        for kw in keywords:
            kw = kw.strip().lower()
            normalized.append(SYNONYMS.get(kw, kw))
        return normalized

    def _load_processed_files(self) -> set:
        if not self.processed_file.exists():
            return set()
        with open(self.processed_file, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}

    def _save_processed_file(self, file_path: str):
        with open(self.processed_file, 'a', encoding='utf-8') as f:
            f.write(f"{file_path}\n")

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, bool]:
        return extract_text_from_pdf_path(pdf_path)

    def normalize_text(self, text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove special characters but keep spaces, numbers, and letters
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text).strip()
        # Replace synonyms
        for abbrev, full in SYNONYMS.items():
            text = text.replace(abbrev, full)
        return text

    def match_keywords(self, text: str) -> Tuple[float, List[str], List[str]]:
        normalized_text = self.normalize_text(text)
        matched_keywords = []
        total_weight = sum(self.weights)
        matched_weight = 0.0
        
        for keyword, weight in zip(self.keywords, self.weights):
            matched = False
            
            # Exact match
            if keyword in normalized_text:
                matched = True
            
            # Fuzzy match if not exact
            if not matched:
                # Check all words in text for fuzzy match
                words = normalized_text.split()
                for word in words:
                    if fuzz.ratio(keyword, word) >= self.fuzzy_threshold:
                        matched = True
                        break
            
            # Semantic match if enabled and not yet matched
            if not matched and self.use_semantic and self.semantic_model:
                text_embedding = self.semantic_model.encode(normalized_text, convert_to_tensor=True)
                cosine_scores = util.cos_sim(text_embedding, self.keyword_embeddings)
                max_score = cosine_scores[0][self.keywords.index(keyword)].item()
                if max_score >= self.semantic_threshold:
                    matched = True
            
            if matched:
                matched_keywords.append(keyword)
                matched_weight += weight
        
        score = matched_weight / total_weight if total_weight > 0 else 0.0
        missing_keywords = [kw for kw in self.keywords if kw not in matched_keywords]
        
        return score, matched_keywords, missing_keywords

    def extract_candidate_name(self, pdf_path: Path, text: str) -> str:
        # Try to extract name from filename first
        filename = pdf_path.stem
        # Remove common resume suffixes
        filename = re.sub(r'[_\-\s]resume', '', filename, flags=re.IGNORECASE)
        filename = re.sub(r'[_\-\s]cv', '', filename, flags=re.IGNORECASE)
        
        # Check if filename has reasonable name (capitalized, 2-4 words)
        filename_parts = [part for part in filename.split() if part.strip()]
        if len(filename_parts) >= 2 and all(part[0].isupper() for part in filename_parts):
            return ' '.join(filename_parts)
        
        # Try to extract from text using common patterns
        lines = text.split('\n')
        for line in lines[:20]:  # Check first 20 lines (name is usually at top)
            line = line.strip()
            if line:
                # Match patterns like "John Doe", "John M. Doe"
                name_match = re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', line)
                if name_match:
                    return name_match.group(0)
        
        return "Unknown"

    def process_resume(self, pdf_path: Path) -> Dict:
        try:
            logger.info(f"Processing {pdf_path.name}...")
            
            # Extract text
            text, is_scanned = self.extract_text_from_pdf(pdf_path)
            
            if len(text.strip()) == 0:
                logger.warning(f"No text could be extracted from {pdf_path.name}")
                return {
                    "file_name": pdf_path.name,
                    "candidate_name": "Unknown",
                    "score": 0.0,
                    "matched_keywords": [],
                    "missing_keywords": self.keywords,
                    "shortlisted": "No",
                    "is_scanned": is_scanned,
                    "error": "No text extracted"
                }
            
            # Extract candidate name
            candidate_name = self.extract_candidate_name(pdf_path, text)
            
            # Match keywords
            score, matched_keywords, missing_keywords = self.match_keywords(text)
            
            # Determine shortlist status
            shortlisted = "Yes" if score >= self.min_threshold else "No"
            
            # Copy to shortlisted folder if shortlisted
            if shortlisted == "Yes":
                dest_path = self.shortlisted_folder / pdf_path.name
                shutil.copy2(pdf_path, dest_path)
                logger.info(f"Shortlisted: {pdf_path.name} (score: {score:.2%})")
            
            return {
                "file_name": pdf_path.name,
                "candidate_name": candidate_name,
                "score": score,
                "matched_keywords": matched_keywords,
                "missing_keywords": missing_keywords,
                "shortlisted": shortlisted,
                "is_scanned": is_scanned,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            return {
                "file_name": pdf_path.name,
                "candidate_name": "Unknown",
                "score": 0.0,
                "matched_keywords": [],
                "missing_keywords": self.keywords,
                "shortlisted": "No",
                "is_scanned": False,
                "error": str(e)
            }

    def process_all_resumes(self) -> pd.DataFrame:
        results = []
        pdf_files = list(self.resumes_folder.rglob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            # Skip already processed files for resumability
            if str(pdf_path) in self.processed_files:
                logger.info(f"Skipping already processed: {pdf_path.name}")
                continue
            
            result = self.process_resume(pdf_path)
            results.append(result)
            self._save_processed_file(str(pdf_path))
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by score descending
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        
        return df

    def generate_report(self, df: pd.DataFrame, output_path: str = "shortlisting_report.xlsx"):
        if df.empty:
            logger.warning("No results to report")
            return
        
        # Prepare data for Excel
        df['score'] = df['score'].apply(lambda x: f"{x:.2%}")
        df['matched_keywords'] = df['matched_keywords'].apply(lambda x: ', '.join(x) if x else '')
        df['missing_keywords'] = df['missing_keywords'].apply(lambda x: ', '.join(x) if x else '')
        
        # Save to Excel
        output_file = self.resumes_folder.parent / output_path
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        logger.info(f"Report generated successfully at {output_file}")
        
        # Generate summary
        logger.info("\n" + "="*50)
        logger.info("SUMMARY REPORT")
        logger.info("="*50)
        logger.info(f"Total resumes processed: {len(df)}")
        logger.info(f"Shortlisted: {len(df[df['shortlisted'] == 'Yes'])}")
        logger.info(f"Rejected: {len(df[df['shortlisted'] == 'No'])}")
        if not df.empty and 'score' in df.columns:
            try:
                avg_score = df['score'].str.rstrip('%').astype(float).mean()
                logger.info(f"Average match score: {avg_score:.2f}%")
                top_candidates = df.head(10)
                logger.info("\nTop 10 Candidates:")
                for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                    logger.info(f"  {i}. {row['candidate_name']} - {row['file_name']} - Score: {row['score']}")
            except:
                pass
        logger.info("="*50 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ATS Resume Shortlisting Tool")
    parser.add_argument("--folder", required=True, help="Path to folder containing resumes")
    parser.add_argument("--keywords", required=True, help="Comma-separated list of keywords")
    parser.add_argument("--weights", help="Comma-separated list of weights (same number as keywords)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Minimum score threshold (default: 0.6 = 60%%)")
    parser.add_argument("--fuzzy", type=float, default=85.0, help="Fuzzy match threshold (0-100, default: 85)")
    parser.add_argument("--semantic", action="store_true", help="Enable semantic similarity matching")
    parser.add_argument("--semantic-threshold", type=float, default=0.7, help="Semantic similarity threshold (0-1, default: 0.7)")
    
    args = parser.parse_args()
    
    # Parse keywords
    keywords = [kw.strip() for kw in args.keywords.split(',') if kw.strip()]
    
    # Parse weights
    weights = None
    if args.weights:
        weights = [float(w.strip()) for w in args.weights.split(',') if w.strip()]
        if len(weights) != len(keywords):
            logger.error("Number of weights must match number of keywords")
            return
    
    # Initialize and run
    shortlister = ATSResumeShortlister(
        resumes_folder=args.folder,
        keywords=keywords,
        weights=weights,
        min_threshold=args.threshold,
        fuzzy_threshold=args.fuzzy,
        use_semantic=args.semantic,
        semantic_threshold=args.semantic_threshold
    )
    
    results_df = shortlister.process_all_resumes()
    shortlister.generate_report(results_df)


if __name__ == "__main__":
    main()

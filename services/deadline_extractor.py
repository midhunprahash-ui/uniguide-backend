"""
Deadline Extractor Service
Extracts deadlines, events, and important dates from circular documents using Gemini AI.
"""
import json
import logging
from datetime import datetime, date
from typing import Optional
import os

from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured for deadline extraction")
else:
    logger.warning("GEMINI_API_KEY not found - deadline extraction will be disabled")

# Extraction prompt for Gemini
EXTRACTION_PROMPT = """Analyze this circular/document text and extract ALL deadlines, events, and important dates.

For each deadline found, provide a JSON object with these fields:
- title: Short descriptive title (max 50 chars)
- description: Full context from the circular (max 200 chars)
- event_type: One of ["payment", "exam", "registration", "event", "submission", "other"]
- deadline_date: Date in YYYY-MM-DD format
- deadline_time: Time in HH:MM format (optional, null if not specified)
- priority: Based on urgency ["critical", "high", "normal", "low"]
- target_streams: Array of stream codes (e.g., ["UG", "PG"]) or [] for all
- target_departments: Array of department codes (e.g., ["CSE", "ECE"]) or [] for all
- target_years: Array of year numbers (e.g., [1, 2, 3, 4]) or [] for all
- confidence: Your confidence in this extraction (0.0 to 1.0)
- extracted_text: The exact text snippet from which you extracted this deadline

Priority guidelines:
- critical: Immediate action required, < 3 days away, severe consequences for missing
- high: Important deadline, financial implications, or exam-related
- normal: Standard academic deadline
- low: Optional events or flexible deadlines

Return ONLY a valid JSON array of deadline objects. If no deadlines found, return an empty array [].

Example output:
[
  {
    "title": "Exam Fee Payment",
    "description": "Pay semester exam fees to the accounts section before the deadline",
    "event_type": "payment",
    "deadline_date": "2026-01-25",
    "deadline_time": "17:00",
    "priority": "high",
    "target_streams": [],
    "target_departments": [],
    "target_years": [],
    "confidence": 0.95,
    "extracted_text": "Exam fees must be paid by 25th January 2026, 5:00 PM"
  }
]

Document text to analyze:
"""


class DeadlineExtractor:
    """
    Extracts deadlines from document text using Gemini AI.
    """
    
    def __init__(self):
        self.model = None
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("DeadlineExtractor initialized with Gemini 2.0 Flash")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
    
    def extract_deadlines(
        self,
        document_text: str,
        document_date: Optional[date] = None
    ) -> list[dict]:
        """
        Extract deadlines from document text.
        
        Args:
            document_text: The full text content of the circular/document
            document_date: Optional date when the document was created (for relative date resolution)
            
        Returns:
            List of extracted deadline dictionaries
        """
        if not self.model:
            logger.warning("Gemini model not available, skipping deadline extraction")
            return []
        
        if not document_text or len(document_text.strip()) < 20:
            logger.debug("Document text too short for deadline extraction")
            return []
        
        try:
            # Add context about current date for relative date resolution
            today = document_date or date.today()
            context = f"\n\nCurrent date for reference: {today.strftime('%Y-%m-%d')}\n\n"
            
            # Call Gemini
            prompt = EXTRACTION_PROMPT + context + document_text[:8000]  # Limit text length
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1  # Low temperature for more consistent extraction
                )
            )
            
            # Parse response
            result_text = response.text.strip()
            
            # Debug: Log the raw response
            logger.info(f"Gemini deadline extraction raw response (first 500 chars): {result_text[:500]}")
            
            # Try to parse JSON
            deadlines = json.loads(result_text)
            
            if not isinstance(deadlines, list):
                logger.warning("Gemini returned non-list response")
                return []
            
            logger.info(f"Gemini extracted {len(deadlines)} deadlines before validation")
            
            # Validate and filter deadlines
            valid_deadlines = []
            for dl in deadlines:
                if self._validate_deadline(dl, today):
                    valid_deadlines.append(dl)
                else:
                    logger.debug(f"Rejected deadline: {dl.get('title', 'unknown')} - date: {dl.get('deadline_date', 'none')}")
            
            logger.info(f"Extracted {len(valid_deadlines)} valid deadlines from document (after filtering)")
            return valid_deadlines
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error during deadline extraction: {e}")
            return []
    
    def _validate_deadline(self, deadline: dict, reference_date: date) -> bool:
        """
        Validate a deadline object.
        
        Args:
            deadline: The deadline dictionary to validate
            reference_date: Reference date for filtering past deadlines
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required = ["title", "event_type", "deadline_date"]
        for field in required:
            if field not in deadline or not deadline[field]:
                logger.debug(f"Deadline missing required field: {field}")
                return False
        
        # Validate date format
        try:
            deadline_date = datetime.strptime(deadline["deadline_date"], "%Y-%m-%d").date()
            
            # Skip past deadlines (more than 1 day ago)
            if deadline_date < reference_date:
                logger.debug(f"Skipping past deadline: {deadline['title']}")
                return False
                
        except ValueError:
            logger.debug(f"Invalid date format: {deadline['deadline_date']}")
            return False
        
        # Validate event_type
        valid_types = ["payment", "exam", "registration", "event", "submission", "other"]
        if deadline.get("event_type") not in valid_types:
            deadline["event_type"] = "other"
        
        # Validate priority
        valid_priorities = ["critical", "high", "normal", "low"]
        if deadline.get("priority") not in valid_priorities:
            deadline["priority"] = "normal"
        
        # Ensure arrays are lists
        for arr_field in ["target_streams", "target_departments", "target_years"]:
            if not isinstance(deadline.get(arr_field), list):
                deadline[arr_field] = []
        
        # Validate confidence
        confidence = deadline.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            deadline["confidence"] = 0.5
        
        return True


# Singleton instance
deadline_extractor = DeadlineExtractor()


def extract_deadlines_from_text(text: str, doc_date: Optional[date] = None) -> list[dict]:
    """
    Convenience function to extract deadlines from text.
    
    Args:
        text: Document text content
        doc_date: Optional document date
        
    Returns:
        List of extracted deadline dictionaries
    """
    return deadline_extractor.extract_deadlines(text, doc_date)

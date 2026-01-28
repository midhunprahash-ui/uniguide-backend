"""
Deadline & Event Extractor Service
Extracts deadlines, events, and important dates from documents using Gemini AI.

Design Principles:
- Single Responsibility: Focus only on extraction logic
- Configurable: Event types and scoring weights are easily adjustable
- Efficient: Uses low temperature for consistent output, limits text input
"""
import json
import logging
from datetime import datetime, date
from typing import Optional
from dataclasses import dataclass
from enum import Enum
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


# ============================================================================
# EVENT TYPE DEFINITIONS
# Clean, extensible enum for categorization
# ============================================================================

class EventType(str, Enum):
    """All supported calendar event types."""
    PAYMENT = "payment"           # Fees, dues, fines
    EXAM = "exam"                 # Tests, vivas, assessments
    REGISTRATION = "registration" # Enrollments, form submissions
    SUBMISSION = "submission"     # Assignments, projects, reports
    EVENT = "event"               # Seminars, workshops, functions
    HOLIDAY = "holiday"           # College holidays, breaks
    ANNOUNCEMENT = "announcement" # Important notices (use doc date if no specific date)
    CLASS = "class"               # Special classes, remedial sessions
    MEETING = "meeting"           # Staff meetings, PTMs
    OTHER = "other"               # Anything else


class Priority(str, Enum):
    """Priority levels for events."""
    CRITICAL = "critical"  # Immediate action, severe consequences
    HIGH = "high"          # Important, financial/academic impact
    NORMAL = "normal"      # Standard deadline
    LOW = "low"            # Optional, flexible


# ============================================================================
# SMART SCORING ALGORITHM
# Eisenhower-inspired: Urgency × Importance
# ============================================================================

# Scoring weights - easily adjustable
EVENT_TYPE_WEIGHTS = {
    EventType.EXAM: 15,
    EventType.PAYMENT: 14,
    EventType.SUBMISSION: 12,
    EventType.REGISTRATION: 10,
    EventType.CLASS: 8,
    EventType.MEETING: 6,
    EventType.EVENT: 5,
    EventType.ANNOUNCEMENT: 4,
    EventType.HOLIDAY: 3,
    EventType.OTHER: 3,
}

PRIORITY_WEIGHTS = {
    Priority.CRITICAL: 25,
    Priority.HIGH: 18,
    Priority.NORMAL: 10,
    Priority.LOW: 5,
}


def calculate_smart_score(
    deadline_date: date,
    event_type: str,
    priority: str,
    reference_date: Optional[date] = None
) -> int:
    """
    Calculate a smart ranking score using Eisenhower Matrix principles.
    
    Score = Urgency (time-based, 0-50) + Importance (type+priority, 0-50)
    Range: 0-100 (higher = more urgent/important)
    
    Args:
        deadline_date: When the event occurs
        event_type: Type of event (exam, payment, etc.)
        priority: Priority level (critical, high, normal, low)
        reference_date: Today's date (defaults to date.today())
    
    Returns:
        Integer score from 0-100
    """
    today = reference_date or date.today()
    days_left = (deadline_date - today).days
    
    # URGENCY SCORE (0-50): Exponential decay based on time remaining
    if days_left <= 0:
        urgency = 50  # Today or overdue
    elif days_left == 1:
        urgency = 45  # Tomorrow
    elif days_left <= 3:
        urgency = 40  # Critical window
    elif days_left <= 7:
        urgency = 30  # This week
    elif days_left <= 14:
        urgency = 20  # Next two weeks
    elif days_left <= 30:
        urgency = 10  # This month
    else:
        urgency = 5   # Far future
    
    # IMPORTANCE SCORE (0-50): Event type + priority
    type_weight = EVENT_TYPE_WEIGHTS.get(
        EventType(event_type) if event_type in [e.value for e in EventType] else EventType.OTHER, 
        3
    )
    priority_weight = PRIORITY_WEIGHTS.get(
        Priority(priority) if priority in [p.value for p in Priority] else Priority.NORMAL,
        10
    )
    importance = min(type_weight + priority_weight, 50)
    
    return urgency + importance


# ============================================================================
# AI EXTRACTION PROMPT
# Comprehensive extraction covering all event types
# ============================================================================

EXTRACTION_PROMPT = """Analyze this college document and extract ALL calendar-worthy items.

Look for:
- DEADLINES: Payment dues, registration closes, submission deadlines
- EXAMS: Tests, vivas, practicals, assessments (often span multiple days!)
- EVENTS: Seminars, workshops, fests, functions, competitions
- HOLIDAYS: College holidays, vacation periods, special leaves (often span multiple days!)
- CLASSES: Remedial classes, extra sessions, lab timings
- MEETINGS: PTMs, staff meetings, orientations
- ANNOUNCEMENTS: Important notices that students should know about

For each item, return a JSON object with:
- title: Clear, concise title (max 60 chars)
- description: Full context (max 250 chars)
- event_type: One of ["payment", "exam", "registration", "submission", "event", "holiday", "announcement", "class", "meeting", "other"]
- deadline_date: START date in YYYY-MM-DD format (when the event begins)
- end_date: END date in YYYY-MM-DD format for multi-day events (null for single-day events)
- deadline_time: Time in HH:MM format (null if not specified)
- priority: Based on urgency ["critical", "high", "normal", "low"]
- target_streams: Array like ["UG", "PG"] or [] for all
- target_departments: Array like ["CSE", "ECE"] or [] for all
- target_years: Array like [1, 2, 3, 4] or [] for all
- confidence: Your confidence 0.0-1.0
- extracted_text: The exact source text (max 200 chars)

IMPORTANT - Multi-day events:
- Exam periods (e.g., "Internal Exams from 5th to 15th Feb") → deadline_date="2026-02-05", end_date="2026-02-15"
- Holidays (e.g., "Semester break from 1st to 10th Jan") → deadline_date="2026-01-01", end_date="2026-01-10"
- Single day events → end_date should be null

Priority guidelines:
- critical: Today/tomorrow, severe consequences (fine, debarment)
- high: Within a week, financial/academic impact
- normal: Standard deadline with buffer time
- low: Optional, informational, or flexible

Return ONLY a valid JSON array. If no items found, return [].

Examples:
[
  {
    "title": "Internal Assessment Exams",
    "description": "Internal exams for all subjects. Refer to schedule for individual dates.",
    "event_type": "exam",
    "deadline_date": "2026-02-05",
    "end_date": "2026-02-15",
    "deadline_time": null,
    "priority": "high",
    "target_streams": [],
    "target_departments": [],
    "target_years": [],
    "confidence": 0.95,
    "extracted_text": "Internal Assessment Exams will be conducted from 5th to 15th February 2026"
  },
  {
    "title": "Semester Exam Fee Payment",
    "description": "Pay exam fees to accounts section. Late fee applies after deadline.",
    "event_type": "payment",
    "deadline_date": "2026-01-25",
    "end_date": null,
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


# ============================================================================
# DEADLINE EXTRACTOR CLASS
# ============================================================================

class DeadlineExtractor:
    """
    Extracts deadlines and events from document text using Gemini AI.
    
    Thread-safe, stateless extraction with configurable validation.
    """
    
    def __init__(self):
        self.model = None
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("DeadlineExtractor initialized with Gemini 2.0 Flash")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
    
    def extract_events(
        self,
        document_text: str,
        document_date: Optional[date] = None,
        include_past: bool = False
    ) -> list[dict]:
        """
        Extract all calendar-worthy events from document text.
        
        Args:
            document_text: Full text content of the document
            document_date: When document was created (for relative date resolution)
            include_past: If True, include past events; if False, filter them out
            
        Returns:
            List of extracted event dictionaries with smart_score added
        """
        if not self.model:
            logger.warning("Gemini model not available, skipping extraction")
            return []
        
        if not document_text or len(document_text.strip()) < 20:
            logger.debug("Document text too short for extraction")
            return []
        
        try:
            # Add context about current/document date
            today = document_date or date.today()
            context = f"\n\nDocument date: {today.strftime('%Y-%m-%d')}\nCurrent date: {date.today().strftime('%Y-%m-%d')}\n\n"
            
            # Limit text to prevent token overflow (approx 12k chars = 3k tokens)
            prompt = EXTRACTION_PROMPT + context + document_text[:12000]
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1  # Low temperature for consistency
                )
            )
            
            result_text = response.text.strip()
            logger.debug(f"AI extraction response length: {len(result_text)} chars")
            
            events = json.loads(result_text)
            
            if not isinstance(events, list):
                logger.warning("AI returned non-list response")
                return []
            
            logger.info(f"AI extracted {len(events)} events before validation")
            
            # Validate and enhance each event
            valid_events = []
            for event in events:
                validated = self._validate_and_enhance(event, today, include_past)
                if validated:
                    valid_events.append(validated)
            
            logger.info(f"Final: {len(valid_events)} valid events after filtering")
            return valid_events
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error during event extraction: {e}")
            return []
    
    def _validate_and_enhance(
        self, 
        event: dict, 
        reference_date: date,
        include_past: bool
    ) -> Optional[dict]:
        """
        Validate an extracted event and add smart_score.
        
        Returns:
            Enhanced event dict or None if invalid
        """
        # Check required fields
        required = ["title", "event_type", "deadline_date"]
        for field in required:
            if field not in event or not event[field]:
                logger.debug(f"Event missing required field: {field}")
                return None
        
        # Validate and parse date
        try:
            event_date = datetime.strptime(event["deadline_date"], "%Y-%m-%d").date()
            
            # Filter past events unless explicitly included
            if not include_past and event_date < reference_date:
                logger.debug(f"Skipping past event: {event['title']}")
                return None
                
        except ValueError:
            logger.debug(f"Invalid date format: {event['deadline_date']}")
            return None
        
        # Normalize event_type
        valid_types = [e.value for e in EventType]
        if event.get("event_type") not in valid_types:
            event["event_type"] = EventType.OTHER.value
        
        # Normalize priority
        valid_priorities = [p.value for p in Priority]
        if event.get("priority") not in valid_priorities:
            event["priority"] = Priority.NORMAL.value
        
        # Ensure arrays are lists
        for field in ["target_streams", "target_departments", "target_years"]:
            if not isinstance(event.get(field), list):
                event[field] = []
        
        # Validate confidence
        confidence = event.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            event["confidence"] = 0.5
        
        # Add smart ranking score
        event["smart_score"] = calculate_smart_score(
            deadline_date=event_date,
            event_type=event["event_type"],
            priority=event["priority"]
        )
        
        return event
    
    # Backwards compatibility alias
    def extract_deadlines(
        self,
        document_text: str,
        document_date: Optional[date] = None
    ) -> list[dict]:
        """Alias for extract_events() for backwards compatibility."""
        return self.extract_events(document_text, document_date)


# ============================================================================
# MODULE-LEVEL SINGLETON AND CONVENIENCE FUNCTIONS
# ============================================================================

# Singleton instance
_extractor_instance: Optional[DeadlineExtractor] = None


def get_extractor() -> DeadlineExtractor:
    """Get or create the singleton DeadlineExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = DeadlineExtractor()
    return _extractor_instance


def extract_deadlines_from_text(text: str, doc_date: Optional[date] = None) -> list[dict]:
    """
    Convenience function to extract events from text.
    
    Args:
        text: Document text content
        doc_date: Optional document date
        
    Returns:
        List of extracted event dictionaries
    """
    return get_extractor().extract_events(text, doc_date)

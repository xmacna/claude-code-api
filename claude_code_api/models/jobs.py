"""Job models for async Brain API."""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# Job status enum
JobStatus = Literal["queued", "processing", "waiting_input", "completed", "failed", "cancelled"]


class ProgressEntry(BaseModel):
    """Progress log entry."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str


class JobRequest(BaseModel):
    """Request to create a new async job."""
    prompt: str = Field(..., description="The prompt to send to Claude")
    model: str = Field(default="sonnet", description="Model to use")
    session_id: Optional[str] = Field(None, description="Session ID to resume conversation")
    callback_url: Optional[str] = Field(None, description="URL to POST callbacks to")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata to pass back in callbacks")
    system_prompt: Optional[str] = Field(None, description="System prompt override")


class JobResponse(BaseModel):
    """Response when creating a job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default="queued", description="Current job status")
    session_id: Optional[str] = Field(None, description="Claude session ID (available after processing starts)")


class JobStatusResponse(BaseModel):
    """Full job status response."""
    job_id: str
    status: JobStatus
    session_id: Optional[str] = None

    # Input
    prompt: str
    model: str

    # Progress tracking
    progress: List[ProgressEntry] = Field(default_factory=list)

    # Output
    output: Optional[str] = None
    error: Optional[str] = None
    waiting_for: Optional[str] = None  # Question from Claude if waiting_input

    # Timestamps
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobContinueRequest(BaseModel):
    """Request to continue a job waiting for input."""
    input: str = Field(..., description="User input to send to Claude")


class CallbackEvent(BaseModel):
    """Callback event sent to callback_url."""
    event: Literal["progress", "waiting_input", "complete", "error", "cancelled"]
    job_id: str
    session_id: Optional[str] = None
    message: Optional[str] = None  # For progress
    question: Optional[str] = None  # For waiting_input
    output: Optional[str] = None  # For complete
    error: Optional[str] = None  # For error
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobListResponse(BaseModel):
    """Response for listing jobs."""
    jobs: List[JobStatusResponse]
    total: int

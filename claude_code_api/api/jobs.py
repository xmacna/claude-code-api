"""Jobs API endpoints for async processing."""

from typing import Optional, List
from fastapi import APIRouter, Request, HTTPException, status, Query
from fastapi.responses import JSONResponse
import structlog

from datetime import datetime
from claude_code_api.models.jobs import (
    JobRequest,
    JobResponse,
    JobStatusResponse,
    JobContinueRequest,
    JobListResponse,
    ProgressEntry,
)

logger = structlog.get_logger()
router = APIRouter()


def _parse_progress(progress_list) -> List[ProgressEntry]:
    """Parse progress entries from job progress list."""
    entries = []
    for p in (progress_list or []):
        ts = p.get("ts")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        entries.append(ProgressEntry(timestamp=ts, message=p.get("msg", "")))
    return entries


def _job_to_response(job) -> JobStatusResponse:
    """Convert Job object to JobStatusResponse."""
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        session_id=job.session_id,
        prompt=job.prompt,
        model=job.model,
        progress=_parse_progress(job.progress),
        output=job.output,
        error=job.error,
        waiting_for=job.waiting_for,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        metadata=job.metadata,
    )


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: JobRequest,
    req: Request
) -> JobResponse:
    """
    Create a new async job.

    Returns immediately with job_id. Results are delivered via callback_url.
    """
    job_manager = req.app.state.job_manager

    try:
        job = await job_manager.create_job(
            prompt=request.prompt,
            model=request.model,
            session_id=request.session_id,
            callback_url=request.callback_url,
            metadata=request.metadata,
            system_prompt=request.system_prompt,
        )

        logger.info(
            "Job created via API",
            job_id=job.job_id,
            model=request.model,
            has_callback=bool(request.callback_url),
        )

        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            session_id=job.session_id,
        )

    except Exception as e:
        logger.error("Failed to create job", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": str(e), "type": "internal_error"}}
        )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    req: Request
) -> JobStatusResponse:
    """Get status of a specific job."""
    job_manager = req.app.state.job_manager

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"message": f"Job {job_id} not found", "type": "not_found"}}
        )

    return _job_to_response(job)


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    req: Request
) -> dict:
    """Cancel a running job."""
    job_manager = req.app.state.job_manager

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"message": f"Job {job_id} not found", "type": "not_found"}}
        )

    success = await job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": f"Cannot cancel job in status {job.status}", "type": "invalid_status"}}
        )

    logger.info("Job cancelled via API", job_id=job_id)

    return {
        "job_id": job_id,
        "status": "cancelled"
    }


@router.post("/jobs/{job_id}/continue")
async def continue_job(
    job_id: str,
    request: JobContinueRequest,
    req: Request
) -> dict:
    """
    Continue a job that is waiting for input.

    Use this when Claude asked a question (status: waiting_input).
    """
    job_manager = req.app.state.job_manager

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"message": f"Job {job_id} not found", "type": "not_found"}}
        )

    if job.status != "waiting_input":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Job is not waiting for input (status: {job.status})",
                    "type": "invalid_status"
                }
            }
        )

    success = await job_manager.continue_job(job_id, request.input)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Failed to continue job", "type": "internal_error"}}
        )

    logger.info("Job continued via API", job_id=job_id)

    return {
        "job_id": job_id,
        "status": "processing"
    }


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    req: Request,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return"),
) -> JobListResponse:
    """List jobs, optionally filtered by status."""
    job_manager = req.app.state.job_manager

    jobs = job_manager.list_jobs(status=status, limit=limit)

    return JobListResponse(
        jobs=[_job_to_response(job) for job in jobs],
        total=len(jobs)
    )

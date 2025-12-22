"""Async job manager for Claude Code processing."""

import asyncio
import json
import os
import re
import uuid
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List
import structlog

from .config import settings
from .claude_manager import create_project_directory

logger = structlog.get_logger()


class Job:
    """Represents a single async job."""

    def __init__(
        self,
        job_id: str,
        prompt: str,
        model: str = "sonnet",
        session_id: Optional[str] = None,
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.job_id = job_id
        self.prompt = prompt
        self.model = model
        self.session_id = session_id
        self.callback_url = callback_url
        self.metadata = metadata or {}
        self.system_prompt = system_prompt

        # State
        self.status = "queued"
        self.progress: List[Dict[str, Any]] = []
        self.output: Optional[str] = None
        self.error: Optional[str] = None
        self.waiting_for: Optional[str] = None

        # Timestamps
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Process management
        self.process: Optional[asyncio.subprocess.Process] = None
        self._cancel_event = asyncio.Event()
        self._last_progress_sent: Optional[datetime] = None
        self._last_activity: str = "Iniciando..."  # Track last meaningful activity for periodic updates

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "model": self.model,
            "progress": self.progress,
            "output": self.output,
            "error": self.error,
            "waiting_for": self.waiting_for,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    def add_progress(self, message: str):
        """Add progress entry."""
        entry = {"ts": datetime.utcnow().isoformat(), "msg": message}
        self.progress.append(entry)
        self.updated_at = datetime.utcnow()
        logger.info("Job progress", job_id=self.job_id, message=message)


class JobManager:
    """Manages async jobs with background workers."""

    def __init__(self, max_workers: int = 5, progress_throttle_seconds: int = 30):
        self.jobs: Dict[str, Job] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.max_workers = max_workers
        self.progress_throttle_seconds = progress_throttle_seconds
        self.workers: List[asyncio.Task] = []
        self._running = False
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Start the job manager and workers."""
        if self._running:
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

        logger.info("JobManager started", workers=self.max_workers)

    async def stop(self):
        """Stop the job manager and all workers."""
        self._running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()

        # Cancel any running jobs
        for job in self.jobs.values():
            if job.process and job.process.returncode is None:
                job.process.terminate()

        logger.info("JobManager stopped")

    async def create_job(
        self,
        prompt: str,
        model: str = "sonnet",
        session_id: Optional[str] = None,
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Job:
        """Create a new job and queue it for processing."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        job = Job(
            job_id=job_id,
            prompt=prompt,
            model=model,
            session_id=session_id,
            callback_url=callback_url,
            metadata=metadata,
            system_prompt=system_prompt,
        )

        self.jobs[job_id] = job
        await self.job_queue.put(job_id)

        logger.info(
            "Job created",
            job_id=job_id,
            model=model,
            has_callback=bool(callback_url),
            has_session=bool(session_id),
        )

        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status not in ("queued", "processing", "waiting_input"):
            return False

        job._cancel_event.set()

        # Kill the process if running
        if job.process and job.process.returncode is None:
            try:
                job.process.terminate()
                await asyncio.wait_for(job.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                job.process.kill()
            except Exception as e:
                logger.error("Error killing process", job_id=job_id, error=str(e))

        job.status = "cancelled"
        job.updated_at = datetime.utcnow()
        job.completed_at = datetime.utcnow()

        # Send callback
        await self._send_callback(job, "cancelled")

        logger.info("Job cancelled", job_id=job_id)
        return True

    async def continue_job(self, job_id: str, user_input: str) -> bool:
        """Continue a job waiting for input."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status != "waiting_input":
            return False

        # Update job
        job.status = "processing"
        job.waiting_for = None
        job.prompt = user_input  # New prompt
        job.updated_at = datetime.utcnow()

        # Re-queue for processing
        await self.job_queue.put(job_id)

        logger.info("Job continued", job_id=job_id)
        return True

    def list_jobs(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def _worker(self, worker_id: int):
        """Background worker that processes jobs from the queue."""
        logger.info("Worker started", worker_id=worker_id)

        while self._running:
            try:
                # Wait for a job with timeout
                try:
                    job_id = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                job = self.jobs.get(job_id)
                if not job or job._cancel_event.is_set():
                    self.job_queue.task_done()
                    continue

                # Process the job
                await self._process_job(job)
                self.job_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error", worker_id=worker_id, error=str(e))

        logger.info("Worker stopped", worker_id=worker_id)

    async def _process_job(self, job: Job):
        """Process a single job."""
        logger.info("Processing job", job_id=job.job_id, model=job.model)

        job.status = "processing"
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        job.add_progress("Iniciando processamento...")

        try:
            # Build Claude command with unbuffered output for real-time streaming
            # Use stdbuf to force line-buffered stdout
            import shutil
            stdbuf_path = shutil.which("stdbuf")

            if stdbuf_path:
                cmd = [stdbuf_path, "-oL", settings.claude_binary_path]
            else:
                cmd = [settings.claude_binary_path]

            cmd.extend(["-p", job.prompt])

            if job.system_prompt:
                cmd.extend(["--system-prompt", job.system_prompt])

            if job.model:
                cmd.extend(["--model", job.model])

            # Resume existing session if provided
            if job.session_id:
                uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
                if re.match(uuid_pattern, job.session_id, re.IGNORECASE):
                    cmd.extend(["--resume", job.session_id])
                    logger.info("Resuming session", session_id=job.session_id)

            cmd.extend([
                "--output-format", "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--include-partial-messages",
                "--strict-mcp-config",
                "--disable-slash-commands"
            ])

            # Create project directory
            project_path = create_project_directory(f"job_{job.job_id}")

            # Run Claude process
            src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

            logger.info("Starting Claude", command=" ".join(cmd))

            job.process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=src_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )

            # Stream output and parse progress
            output_lines = []
            claude_session_id = None

            line_count = 0
            last_periodic_update = datetime.utcnow()
            periodic_update_interval = 60  # Send progress every 60 seconds

            while True:
                if job._cancel_event.is_set():
                    job.process.terminate()
                    break

                # Check periodic progress before reading (runs every iteration)
                now = datetime.utcnow()
                elapsed = (now - last_periodic_update).total_seconds()
                if elapsed >= periodic_update_interval:
                    last_periodic_update = now
                    elapsed_total = (now - job.started_at).total_seconds() if job.started_at else 0
                    minutes = int(elapsed_total // 60)
                    # Use last meaningful activity instead of generic message
                    msg = f"[{minutes}min] {job._last_activity}" if minutes > 0 else job._last_activity
                    job.add_progress(msg)  # Add to job.progress for polling
                    if job.callback_url:
                        await self._send_callback(job, "progress", message=msg)
                    logger.info("Sent periodic progress", job_id=job.job_id, elapsed_minutes=minutes, activity=job._last_activity)

                # Read line with timeout
                try:
                    line = await asyncio.wait_for(
                        job.process.stdout.readline(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check if process is still running
                    if job.process.returncode is not None:
                        logger.info("Process finished", job_id=job.job_id, lines_read=line_count)
                        break
                    continue

                if not line:
                    logger.info("EOF reached", job_id=job.job_id, lines_read=line_count)
                    break

                line = line.decode().strip()
                if not line:
                    continue

                line_count += 1
                output_lines.append(line)

                # Parse JSON message
                try:
                    data = json.loads(line)

                    # Extract session ID
                    if not claude_session_id and data.get("session_id"):
                        claude_session_id = data["session_id"]
                        job.session_id = claude_session_id
                        logger.info("Got session ID", session_id=claude_session_id)

                    # Log message type for debugging
                    msg_type = data.get("type", "unknown")
                    logger.debug("Claude message", job_id=job.job_id, line=line_count, msg_type=msg_type)

                    # Parse progress from different message types
                    await self._parse_progress(job, data)

                except json.JSONDecodeError:
                    # Non-JSON output
                    logger.debug("Non-JSON output", line=line[:100])

            # Wait for process to complete
            await job.process.wait()

            # Read any remaining stderr
            stderr = await job.process.stderr.read()
            if stderr:
                logger.info("Claude stderr", stderr=stderr.decode()[:500])

            if job._cancel_event.is_set():
                return  # Already handled in cancel_job

            # Process completed
            if job.process.returncode == 0:
                # Output was already built incrementally in _parse_progress
                # Extract any remaining output as fallback
                if not job.output:
                    job.output = self._extract_output(output_lines)
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.add_progress("Processamento concluído")
                await self._send_callback(job, "complete")

            else:
                # Process failed
                job.error = stderr.decode() if stderr else "Unknown error"
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                job.add_progress(f"Erro: {job.error[:100]}")
                await self._send_callback(job, "error")

            job.updated_at = datetime.utcnow()

        except Exception as e:
            logger.error("Job processing error", job_id=job.job_id, error=str(e))
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()
            await self._send_callback(job, "error")

    async def _parse_progress(self, job: Job, data: Dict[str, Any]):
        """Parse progress from Claude output and send callbacks."""
        msg_type = data.get("type")

        # System messages often contain progress info
        if msg_type == "system":
            message = data.get("message", "")
            if message:
                job._last_activity = message[:80]  # Track for periodic updates
                job.add_progress(message)
                await self._maybe_send_progress_callback(job, message)

        # Stream events contain tool use info
        elif msg_type == "stream_event":
            event = data.get("event", {})
            event_type = event.get("type", "")

            # Tool use starts in content_block_start
            if event_type == "content_block_start":
                content_block = event.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    tool_name = content_block.get("name", "tool")
                    activity = f"Usando {tool_name}"
                    job._last_activity = activity
                    job.add_progress(activity)
                    await self._maybe_send_progress_callback(job, f"{activity}...")

            # Content block delta for text streaming
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    job._last_activity = "Escrevendo resposta..."
                    partial_text = delta.get("text", "")
                    if partial_text:
                        if job.output:
                            job.output += partial_text
                        else:
                            job.output = partial_text
                        job.updated_at = datetime.utcnow()

        # Legacy tool_use type (fallback)
        elif msg_type == "tool_use":
            tool_name = data.get("name", "tool")
            activity = f"Usando {tool_name}"
            job._last_activity = activity
            job.add_progress(activity)
            await self._maybe_send_progress_callback(job, f"{activity}...")

        # Tool result
        elif msg_type == "tool_result":
            job.add_progress("Ferramenta executada")

        # Assistant message - capture text and check for questions
        elif msg_type == "assistant":
            content = data.get("message", {}).get("content", [])
            text = self._extract_text_from_content(content)
            if text:
                # Update output incrementally (may already have partial text)
                if job.output:
                    # Only append if not already captured via partial messages
                    if not job.output.endswith(text):
                        job.output += "\n\n" + text
                else:
                    job.output = text
                job.updated_at = datetime.utcnow()

                # Check if Claude is asking a question
                if text.strip().endswith("?"):
                    job.status = "waiting_input"
                    job.waiting_for = text
                    await self._send_callback(job, "waiting_input")

    async def _maybe_send_progress_callback(self, job: Job, message: str):
        """Send progress callback with throttling."""
        now = datetime.utcnow()

        # Throttle progress updates
        if job._last_progress_sent:
            elapsed = (now - job._last_progress_sent).total_seconds()
            if elapsed < self.progress_throttle_seconds:
                return

        job._last_progress_sent = now
        await self._send_callback(job, "progress", message=message)

    async def _send_callback(
        self,
        job: Job,
        event: str,
        message: Optional[str] = None
    ):
        """Send callback to callback_url."""
        if not job.callback_url or not self._http_client:
            return

        payload = {
            "event": event,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "metadata": job.metadata,
        }

        if event == "progress":
            payload["message"] = message
        elif event == "waiting_input":
            payload["question"] = job.waiting_for
        elif event == "complete":
            payload["output"] = job.output
        elif event == "error":
            payload["error"] = job.error

        try:
            response = await self._http_client.post(
                job.callback_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            logger.info(
                "Callback sent",
                job_id=job.job_id,
                callback_event=event,
                status_code=response.status_code
            )
        except Exception as e:
            logger.error(
                "Callback failed",
                job_id=job.job_id,
                callback_event=event,
                error=str(e)
            )

    def _extract_output(self, lines: List[str]) -> str:
        """Extract final text output from Claude messages."""
        output_parts = []

        for line in lines:
            try:
                data = json.loads(line)
                if data.get("type") == "assistant":
                    content = data.get("message", {}).get("content", [])
                    text = self._extract_text_from_content(content)
                    if text:
                        output_parts.append(text)
                elif data.get("type") == "result":
                    # Final result message
                    result = data.get("result", "")
                    if result:
                        output_parts.append(result)
            except json.JSONDecodeError:
                continue

        return "\n\n".join(output_parts) if output_parts else "Processamento concluído sem output."

    def _extract_text_from_content(self, content) -> str:
        """Extract text from Claude content blocks."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts)

        return ""

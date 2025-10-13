#!/usr/bin/env python3
"""
CodeNinja Cursor CLI Bridge Service
A FastAPI service that receives prompts from Laravel and executes them using Cursor CLI
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeNinja Cursor CLI Bridge",
    description="Bridge service between Laravel and Cursor CLI",
    version="1.0.0",
)

# Allow cross-origin requests from the app running on localhost (Laravel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CursorTask(BaseModel):
    """Model for Cursor CLI tasks"""

    project_id: int
    project_name: str
    task_description: str
    working_directory: Optional[str] = None
    organization_name: Optional[str] = None


class CursorResponse(BaseModel):
    """Model for Cursor CLI responses"""

    success: bool
    output: str
    error: Optional[str] = None
    project_path: Optional[str] = None


def get_downloads_path() -> str:
    """Get the user's Downloads directory path"""
    home = os.path.expanduser("~")
    downloads = os.path.join(home, "Downloads")
    return downloads


def find_cursor_agent() -> str:
    """Find the cursor-agent binary path"""
    possible_paths = [
        os.path.expanduser("~/.local/bin/cursor-agent"),
        "/usr/local/bin/cursor-agent",
        "/usr/bin/cursor-agent",
    ]

    for path in possible_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Try to find it in PATH
    try:
        result = subprocess.run(
            ["which", "cursor-agent"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    raise FileNotFoundError("cursor-agent not found. Please install Cursor CLI.")


def execute_cursor_task(task: CursorTask) -> CursorResponse:
    """Execute a task using Cursor CLI"""
    try:
        # Get the working directory
        downloads_path = get_downloads_path()
        working_dir = task.working_directory or downloads_path

        # Find existing project or use downloads directory
        if task.project_id:
            # Look for existing project directories
            project_pattern = f"task-app-{task.project_id}-*"
            project_dirs = list(Path(working_dir).glob(project_pattern))

            if project_dirs:
                # Use the latest project directory
                latest_project = max(project_dirs, key=os.path.getctime)
                working_dir = str(latest_project)
                logger.info(f"Found existing project: {latest_project}")
            else:
                logger.info(
                    f"No existing project found for ID {task.project_id}, using Downloads directory"
                )

        # Ensure working directory exists
        os.makedirs(working_dir, exist_ok=True)

        # Find cursor-agent
        cursor_agent_path = find_cursor_agent()
        logger.info(f"Using cursor-agent at: {cursor_agent_path}")

        # Prepare the command: pass the user's text verbatim
        prompt = task.task_description

        # Change to the working directory and execute cursor-agent
        cmd = [cursor_agent_path, "agent", "--print", prompt]

        logger.info(f"Executing command: {' '.join(cmd)}")
        logger.info(f"Working directory: {working_dir}")

        # Execute the command
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        return CursorResponse(
            success=result.returncode == 0,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None,
            project_path=working_dir,
        )

    except subprocess.TimeoutExpired:
        return CursorResponse(
            success=False, output="", error="Command timed out after 5 minutes"
        )
    except FileNotFoundError as e:
        return CursorResponse(
            success=False, output="", error=f"Cursor CLI not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error executing cursor task: {str(e)}")
        return CursorResponse(
            success=False, output="", error=f"Unexpected error: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "CodeNinja Cursor CLI Bridge",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        cursor_agent_path = find_cursor_agent()
        downloads_path = get_downloads_path()

        return {
            "status": "healthy",
            "cursor_agent_path": cursor_agent_path,
            "downloads_path": downloads_path,
            "downloads_exists": os.path.exists(downloads_path),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/execute", response_model=CursorResponse)
async def execute_cursor_command(task: CursorTask):
    """Execute a Cursor CLI command"""
    logger.info(f"Received task: {task.task_description}")

    try:
        result = execute_cursor_task(task)
        logger.info(f"Task completed with success: {result.success}")
        return result
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute-stream")
async def execute_cursor_command_stream(task: CursorTask):
    """Execute a Cursor CLI command with real-time streaming output"""
    logger.info(f"Received streaming task: {task.task_description}")

    try:
        from fastapi.responses import StreamingResponse
        import sys

        def generate_output():
            try:
                downloads_path = get_downloads_path()
                working_dir = task.working_directory or downloads_path

                # Find existing project or use downloads directory
                latest_project_path = None
                if task.project_id:
                    # Look for existing project directories
                    project_pattern = f"task-app-{task.project_id}-*"
                    project_dirs = list(Path(working_dir).glob(project_pattern))

                    if project_dirs:
                        # Use the latest project directory
                        latest_project_path = max(project_dirs, key=os.path.getctime)
                        working_dir = str(latest_project_path)
                        logger.info(f"Found existing project: {latest_project_path}")
                    else:
                        logger.info(
                            f"No existing project found for ID {task.project_id}, using Downloads directory"
                        )

                # Ensure working directory exists
                os.makedirs(working_dir, exist_ok=True)

                # Find cursor-agent
                cursor_agent_path = find_cursor_agent()
                logger.info(f"Using cursor-agent at: {cursor_agent_path}")

                # Prepare the command: pass the user's text verbatim
                prompt = task.task_description

                # Change to the working directory and execute cursor-agent with streaming
                cmd = [
                    cursor_agent_path,
                    "agent",
                    "--print",
                    "--output-format",
                    "stream-json",
                    "--stream-partial-output",
                    prompt,
                ]

                logger.info(f"Executing command: {' '.join(cmd)}")
                logger.info(f"Working directory: {working_dir}")
                print(f"COMMAND: {' '.join(cmd)}", flush=True)  # Debug print
                print(f"PROMPT: {prompt}", flush=True)  # Debug print

                # Execute the command with real-time streaming output
                process = subprocess.Popen(
                    cmd,
                    cwd=working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=0,  # Unbuffered for real-time output
                    universal_newlines=True,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                    },  # Force unbuffered output
                )

                print(f"PROCESS STARTED: {process.pid}", flush=True)  # Debug print

                # Stream output line by line for JSON streaming
                print("STARTING STREAMING LOOP", flush=True)  # Debug print
                while True:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            print(
                                f"STREAMING LINE: {repr(line)}", flush=True
                            )  # Debug print

                            # Try to parse as JSON and extract content
                            try:
                                import json

                                data = json.loads(line)

                                # Handle assistant message content
                                if (
                                    data.get("type") == "assistant"
                                    and "message" in data
                                ):
                                    message = data["message"]
                                    if "content" in message and isinstance(
                                        message["content"], list
                                    ):
                                        for content_item in message["content"]:
                                            if (
                                                isinstance(content_item, dict)
                                                and "text" in content_item
                                            ):
                                                text = content_item["text"]
                                                if text:
                                                    yield text
                                                    sys.stdout.flush()  # Force immediate output

                                # Handle other content types
                                elif "content" in data:
                                    content = data["content"]
                                    if content:
                                        yield content
                                        sys.stdout.flush()
                                elif "delta" in data:
                                    delta = data["delta"]
                                    if delta:
                                        yield delta
                                        sys.stdout.flush()
                            except json.JSONDecodeError:
                                # If not JSON, yield the line as-is
                                yield line + "\n"
                                sys.stdout.flush()
                    else:
                        # Check if process is still running
                        if process.poll() is not None:
                            print("PROCESS FINISHED", flush=True)  # Debug print
                            break
                        # No sleep - read immediately for real-time streaming

                # Wait for process to complete
                process.wait()

                print(
                    f"PROCESS COMPLETED: returncode={process.returncode}", flush=True
                )  # Debug print

                if process.returncode == 0:
                    yield "\n\n✅ Task completed successfully!\n"
                else:
                    yield f"\n\n❌ Task failed with exit code: {process.returncode}\n"

            except Exception as e:
                logger.error(f"Error in streaming execution: {str(e)}")
                yield f"\n\n❌ Error: {str(e)}\n"

        return StreamingResponse(
            generate_output(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    except Exception as e:
        logger.error(f"Error in /execute-stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}")
async def get_project_info(project_id: int):
    """Get information about a project"""
    downloads_path = get_downloads_path()
    project_pattern = f"task-app-{project_id}-*"
    project_dirs = list(Path(downloads_path).glob(project_pattern))

    if not project_dirs:
        return {
            "project_id": project_id,
            "exists": False,
            "message": "No projects found",
        }

    # Get the latest project
    latest_project = max(project_dirs, key=os.path.getctime)

    return {
        "project_id": project_id,
        "exists": True,
        "latest_project": str(latest_project),
        "all_projects": [str(p) for p in project_dirs],
        "files": list(latest_project.iterdir()) if latest_project.is_dir() else [],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4567, reload=True, log_level="info")

#!/usr/bin/env python3
"""
CodeNinja Cursor CLI Bridge Service
A FastAPI service that receives prompts from Laravel and executes them using Cursor CLI
"""

import os
import subprocess
import json
import logging
import time
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


class AgentInfo(BaseModel):
    """Model for agent information"""

    id: int
    name: str
    goal: str
    responsibilities: str
    context: Optional[Dict[str, Any]] = None
    capabilities: Optional[list] = None
    command_content: str


class CursorTask(BaseModel):
    """Model for Cursor CLI tasks"""

    project_id: int
    project_name: str
    task_description: str
    working_directory: Optional[str] = None
    workflow_repository_path: Optional[str] = None
    organization_name: Optional[str] = None
    agents: Optional[list[AgentInfo]] = None


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
    logger.info(f"=== FASTAPI RECEIVED PAYLOAD ===")
    logger.info(f"Task description: {task.task_description}")
    logger.info(f"Task agents: {task.agents}")
    logger.info(f"Task project_id: {task.project_id}")
    logger.info(
        f"Task organization_id: {getattr(task, 'organization_id', 'Not provided')}"
    )
    logger.info(f"Task project_name: {getattr(task, 'project_name', 'Not provided')}")
    logger.info(
        f"Task working_directory: {getattr(task, 'working_directory', 'Not provided')}"
    )
    logger.info(f"=== END FASTAPI PAYLOAD ===")

    try:
        from fastapi.responses import StreamingResponse
        import sys

        def generate_output():
            try:
                # Priority: workflow_repository_path > working_directory > Downloads
                if task.workflow_repository_path:
                    working_dir = task.workflow_repository_path
                    logger.info(f"Using workflow repository path: {working_dir}")

                    # Check if the workflow repository directory exists
                    if not os.path.exists(working_dir):
                        error_msg = f"Workflow repository directory does not exist: {working_dir}"
                        logger.error(error_msg)
                        yield f"\n\n❌ Error: {error_msg}\n"
                        yield "Please ensure the GitHub repository has been cloned to the specified path.\n"
                        return

                    if not os.path.isdir(working_dir):
                        error_msg = f"Workflow repository path is not a directory: {working_dir}"
                        logger.error(error_msg)
                        yield f"\n\n❌ Error: {error_msg}\n"
                        return
                else:
                    # Fallback to old behavior for backward compatibility
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
                            latest_project_path = max(
                                project_dirs, key=os.path.getctime
                            )
                            working_dir = str(latest_project_path)
                            logger.info(
                                f"Found existing project: {latest_project_path}"
                            )
                        else:
                            logger.info(
                                f"No existing project found for ID {task.project_id}, using Downloads directory"
                            )

                    # Ensure working directory exists
                    os.makedirs(working_dir, exist_ok=True)

                # Find cursor-agent
                cursor_agent_path = find_cursor_agent()
                logger.info(f"Using cursor-agent at: {cursor_agent_path}")

                # Use the task description directly - agent paths are already converted
                prompt = task.task_description
                logger.info(f"=== FINAL COMMAND TO CURSOR-AGENT ===")
                logger.info(f"Full prompt: {prompt}")
                logger.info(f"=== END COMMAND ===")

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
                logger.info(
                    f"Command working directory contents: {list(Path(working_dir).iterdir()) if Path(working_dir).exists() else 'Directory does not exist'}"
                )
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
                seen_content = set()  # Track seen content to avoid duplication
                complete_response_received = (
                    False  # Track if we've received a complete response
                )
                collected_content = []  # Collect all content chunks
                last_output = ""  # Track last output to detect duplicates
                current_response = ""  # Track current complete response
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

                                # Skip result type messages to avoid duplication
                                if data.get("type") == "result":
                                    continue

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

                                                # Only stream individual chunks, not complete responses
                                                # Complete responses are typically very long (> 500 chars)
                                                # and contain all the chunks we've already streamed
                                                if len(text) > 500:
                                                    # Skip complete responses - we've already streamed the chunks
                                                    print(
                                                        f"SKIPPING LONG TEXT: {len(text)} chars",
                                                        flush=True,
                                                    )
                                                    continue

                                                # Only output if we haven't seen this exact text before
                                                if text not in seen_content:
                                                    print(
                                                        f"STREAMING NEW TEXT: '{text[:50]}...' ({len(text)} chars)",
                                                        flush=True,
                                                    )
                                                    seen_content.add(text)
                                                    yield text
                                                    sys.stdout.flush()
                                                else:
                                                    print(
                                                        f"SKIPPING DUPLICATE: '{text[:50]}...' ({len(text)} chars)",
                                                        flush=True,
                                                    )

                                # Handle other content types
                                elif "content" in data:
                                    content = data["content"]
                                    if content and content not in seen_content:
                                        seen_content.add(content)
                                        yield content
                                        sys.stdout.flush()
                                elif "delta" in data:
                                    delta = data["delta"]
                                    if delta and delta not in seen_content:
                                        seen_content.add(delta)
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


@app.post("/create-agent")
async def create_agent(
    agent: AgentInfo,
    project_id: int = None,
    organization_id: int = None,
    team_member_id: int = None,
):
    """Create a new agent and store its command file in project directory"""
    logger.info(f"Creating agent: {agent.name}")

    try:
        # Determine project directory
        if project_id:
            # Look for existing project directories
            downloads_path = get_downloads_path()
            project_pattern = f"task-app-{project_id}-*"
            project_dirs = list(Path(downloads_path).glob(project_pattern))

            if project_dirs:
                # Use the latest project directory
                project_dir = max(project_dirs, key=os.path.getctime)
            else:
                # Create new project directory
                project_dir = (
                    Path(downloads_path) / f"task-app-{project_id}-{int(time.time())}"
                )
                project_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use test project directory
            project_dir = Path("/home/krunaldodiya/Downloads/test-agent-project")
            project_dir.mkdir(parents=True, exist_ok=True)

        # Create global .codeninja/{organization_id}/{project_id}/agents directory
        if organization_id and project_id:
            # Use global .codeninja directory in user's home directory
            global_codeninja_dir = Path.home() / ".codeninja"
            agents_dir = (
                global_codeninja_dir / str(organization_id) / str(project_id) / "agents"
            )
        else:
            # Fallback to .cursor/commands for backward compatibility
            agents_dir = project_dir / ".cursor" / "commands"

        agents_dir.mkdir(parents=True, exist_ok=True)

        # Create agent file with clean name
        agent_filename = agent.name.lower().replace(" ", "_").replace("-", "_") + ".md"
        agent_file_path = agents_dir / agent_filename

        # Write agent command file
        with open(agent_file_path, "w") as f:
            f.write(agent.command_content)

        logger.info(f"Agent command file created: {agent_file_path}")

        return {
            "success": True,
            "message": f"Agent {agent.name} created successfully",
            "command_file_path": str(agent_file_path),
            "project_directory": str(project_dir),
            "agents_directory": str(agents_dir),
        }

    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List all available agents"""
    try:
        agents_dir = Path(
            "/home/krunaldodiya/WorkSpace/Code/codeninja/storage/app/agents"
        )

        if not agents_dir.exists():
            return {"agents": []}

        agents = []
        for agent_file in agents_dir.glob("*.md"):
            with open(agent_file, "r") as f:
                content = f.read()
                agents.append(
                    {
                        "name": agent_file.stem,
                        "file_path": str(agent_file),
                        "content_preview": (
                            content[:200] + "..." if len(content) > 200 else content
                        ),
                    }
                )

        return {"agents": agents}

    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4567, reload=True, log_level="info")

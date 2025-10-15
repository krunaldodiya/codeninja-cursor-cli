#!/usr/bin/env python3
"""
CodeNinja Cursor CLI Bridge Service
A FastAPI service that receives prompts from Laravel and executes them using Cursor CLI
"""

import os
import subprocess
import logging
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uvicorn import run

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


class GitHubRepository(BaseModel):
    """Model for GitHub repository information"""

    repository_owner: str
    repository_name: str
    repository_full_name: str
    repository_access_token: str
    git_clone_ssh_url: Optional[str] = None


class TestAgentRequest(BaseModel):
    """Model for agent execution requests"""

    agent: str
    task: str
    prompt: str


def get_working_directory() -> str:
    """Get the current working directory path"""
    return os.getcwd()


def get_workflow_working_directory(project_id: int, workflow_id: int) -> str:
    """Get working directory from config.json for a specific workflow"""
    try:
        import json

        codeninja_dir = Path.home() / ".codeninja"
        config_file_path = codeninja_dir / "config.json"

        if not config_file_path.exists():
            logger.info("No config file found, using current directory")
            return os.getcwd()

        with open(config_file_path, "r") as f:
            config = json.load(f)

        workflow_key = f"project_{project_id}_workflow_{workflow_id}"
        if workflow_key not in config:
            logger.info(
                f"No config found for workflow {workflow_key}, using current directory"
            )
            return os.getcwd()

        workflow_config = config[workflow_key]
        working_dir = workflow_config.get("working_directory")

        if not working_dir or not os.path.exists(working_dir):
            logger.warning(
                f"Config found but directory doesn't exist: {working_dir}, using current directory"
            )
            return os.getcwd()

        logger.info(f"Found workflow directory from config: {working_dir}")

        # Update last_accessed timestamp
        config[workflow_key]["last_accessed"] = time.time()
        try:
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to update config timestamp: {e}")

        return working_dir

    except Exception as e:
        logger.warning(f"Error reading config file: {e}, using current directory")
        return os.getcwd()


def generate_ssh_url(repository_full_name: str, repository_access_token: str) -> str:
    """Generate SSH URL with token from repository full name"""
    # Convert full name to SSH format with token
    # Format: git@github.com:username/repo.git
    # With token: https://token@github.com/username/repo.git
    return f"https://{repository_access_token}@github.com/{repository_full_name}.git"


def get_codeninja_repo_path(repo_name: str) -> str:
    """Get the CodeNinja repository path"""
    home = os.path.expanduser("~")
    return os.path.join(home, ".codeninja", repo_name)


def validate_git_repository(repo_path: str) -> bool:
    """Validate that the path is a proper git repository"""
    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        return False

    git_path = os.path.join(repo_path, ".git")
    if not os.path.exists(git_path):
        return False

    # Additional validation: check if git commands work
    try:
        result = subprocess.run(
            ["git", "status"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def clone_repository_if_needed(repo_name: str, ssh_url: str) -> str:
    """Clone repository if it doesn't exist in ~/.codeninja/{repo_name}"""
    repo_path = get_codeninja_repo_path(repo_name)

    # Check if repository already exists and is valid
    if validate_git_repository(repo_path):
        logger.info(f"Valid git repository already exists at: {repo_path}")
        return repo_path
    elif os.path.exists(repo_path):
        logger.warning(
            f"Directory exists but is not a valid git repository: {repo_path}"
        )
        # Remove the directory and clone fresh
        shutil.rmtree(repo_path)

    # Create .codeninja directory if it doesn't exist
    codeninja_dir = os.path.dirname(repo_path)
    os.makedirs(codeninja_dir, exist_ok=True)

    # Clone the repository
    logger.info(f"Cloning repository from: {ssh_url}")
    logger.info(f"To directory: {repo_path}")

    try:
        result = subprocess.run(
            ["git", "clone", ssh_url, repo_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            # Validate the cloned repository
            if validate_git_repository(repo_path):
                logger.info(
                    f"Successfully cloned and validated repository at: {repo_path}"
                )
                return repo_path
            else:
                logger.error(f"Cloned repository failed validation: {repo_path}")
                raise Exception("Cloned repository is not valid")
        else:
            logger.error(f"Failed to clone repository: {result.stderr}")
            raise Exception(f"Git clone failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out after 5 minutes")
        raise Exception("Git clone timed out")
    except Exception as e:
        logger.error(f"Error cloning repository: {str(e)}")
        raise Exception(f"Error cloning repository: {str(e)}")


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
        current_dir = os.getcwd()

        return {
            "status": "healthy",
            "cursor_agent_path": cursor_agent_path,
            "current_directory": current_dir,
            "current_dir_exists": os.path.exists(current_dir),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/projects/{project_id}/workflows/{workflow_id}/execute")
async def execute_workflow_command(
    project_id: int, workflow_id: int, request: TestAgentRequest
):
    """Execute a Cursor CLI command with agent context and streaming response for specific workflow"""
    logger.info(f"Executing agent with goal: {request.agent}")
    logger.info(f"Task: {request.task}")
    logger.info(f"Prompt: {request.prompt}")
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Workflow ID: {workflow_id}")

    try:
        from fastapi.responses import StreamingResponse
        import sys
        import json

        def generate_output():
            try:
                # Determine working directory - always use config.json with path parameters
                # Use config.json to get working directory for the specific workflow
                working_dir = get_workflow_working_directory(project_id, workflow_id)

                # Find cursor-agent
                cursor_agent_path = find_cursor_agent()
                logger.info(f"Using cursor-agent at: {cursor_agent_path}")

                # Create the full prompt with agent context and working directory info
                context_parts = [f"Agent Context: {request.agent}"]
                context_parts.append(f"Current Working Directory: {working_dir}")
                context_parts.append(f"Project ID: {project_id}")
                context_parts.append(f"Workflow ID: {workflow_id}")
                context_parts.append("Repository Type: Project Repository")

                context = "\n".join(context_parts)

                full_prompt = f"""{context}

Task: {request.task}

Prompt: {request.prompt}

Please respond as the agent described in the context above."""

                logger.info(f"Full prompt: {full_prompt}")

                # Execute cursor-agent with streaming output
                cmd = [
                    cursor_agent_path,
                    "agent",
                    "--print",
                    "--output-format",
                    "stream-json",
                    "--stream-partial-output",
                    full_prompt,
                ]

                logger.info(f"Executing command: {' '.join(cmd)}")
                logger.info(f"Working directory: {working_dir}")

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

                logger.info(f"Process started with PID: {process.pid}")

                # Stream output line by line for JSON streaming
                seen_content = set()  # Track seen content to avoid duplication

                while True:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            logger.debug(f"Streaming line: {repr(line)}")

                            # Try to parse as JSON and extract content
                            try:
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
                                                if len(text) > 500:
                                                    continue

                                                # Only output if we haven't seen this exact text before
                                                if text not in seen_content:
                                                    seen_content.add(text)
                                                    yield text
                                                    sys.stdout.flush()

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
                            logger.info("Process finished")
                            break

                # Wait for process to complete
                process.wait()

                logger.info(f"Process completed with return code: {process.returncode}")

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
        logger.error(f"Error in /execute endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_id}/workflows/{workflow_id}/initialize")
async def initialize_workflow(
    project_id: int, workflow_id: int, github_repository: GitHubRepository
):
    """Initialize a workflow by cloning GitHub repository and managing config"""
    logger.info(
        f"Initializing workflow {workflow_id} for project {project_id} with GitHub repository: {github_repository.repository_name}"
    )

    try:
        import json

        # Get config file path
        codeninja_dir = Path.home() / ".codeninja"
        config_file_path = codeninja_dir / "config.json"

        # Ensure .codeninja directory exists
        codeninja_dir.mkdir(exist_ok=True)

        # Load existing config or create new one
        config = {}
        if config_file_path.exists():
            try:
                with open(config_file_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded existing config from: {config_file_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file: {e}, creating new config")
                config = {}

        # Create workflow key
        workflow_key = f"project_{project_id}_workflow_{workflow_id}"

        # Check if workflow is already initialized
        if workflow_key in config:
            existing_config = config[workflow_key]
            existing_working_dir = existing_config.get("working_directory")
            existing_repo_name = existing_config.get("repository_name")

            # Verify if the existing directory exists and matches
            if (
                existing_working_dir
                and os.path.exists(existing_working_dir)
                and existing_repo_name == github_repository.repository_name
            ):

                logger.info(
                    f"Workflow {workflow_id} already initialized with matching repository"
                )
                return {
                    "success": True,
                    "project_id": project_id,
                    "workflow_id": workflow_id,
                    "repository_path": existing_working_dir,
                    "repository_name": github_repository.repository_name,
                    "repository_owner": github_repository.repository_owner,
                    "message": f"Workflow {workflow_id} for project {project_id} already initialized with repository {github_repository.repository_name}",
                    "already_initialized": True,
                }
            else:
                logger.info(
                    f"Existing workflow config found but directory/repo mismatch, reinitializing"
                )

        # Generate SSH URL with token
        ssh_url = generate_ssh_url(
            github_repository.repository_full_name,
            github_repository.repository_access_token,
        )
        logger.info(f"Generated SSH URL: {ssh_url}")

        # Clone repository if needed
        working_dir = clone_repository_if_needed(
            github_repository.repository_name, ssh_url
        )
        logger.info(f"Repository cloned to: {working_dir}")

        # Validate the repository
        if validate_git_repository(working_dir):
            # Update config with new workflow information
            config[workflow_key] = {
                "project_id": project_id,
                "workflow_id": workflow_id,
                "working_directory": working_dir,
                "repository_name": github_repository.repository_name,
                "repository_owner": github_repository.repository_owner,
                "repository_full_name": github_repository.repository_full_name,
                "initialized_at": time.time(),
                "last_accessed": time.time(),
            }

            # Save config file
            try:
                with open(config_file_path, "w") as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Config saved to: {config_file_path}")
            except IOError as e:
                logger.error(f"Failed to save config file: {e}")
                # Continue execution even if config save fails

            return {
                "success": True,
                "project_id": project_id,
                "workflow_id": workflow_id,
                "repository_path": working_dir,
                "repository_name": github_repository.repository_name,
                "repository_owner": github_repository.repository_owner,
                "message": f"Workflow {workflow_id} for project {project_id} initialized successfully with repository {github_repository.repository_name}",
                "already_initialized": False,
            }
        else:
            raise Exception(f"Repository validation failed for: {working_dir}")

    except Exception as e:
        logger.error(
            f"Error initializing workflow {workflow_id} for project {project_id}: {str(e)}"
        )
        return {
            "success": False,
            "project_id": project_id,
            "workflow_id": workflow_id,
            "error": str(e),
            "message": f"Failed to initialize workflow {workflow_id} for project {project_id}",
        }


if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=4567, reload=True, log_level="info")

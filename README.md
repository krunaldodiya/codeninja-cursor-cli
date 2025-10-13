# CodeNinja Cursor CLI Bridge

A FastAPI service that acts as a bridge between Laravel and Cursor CLI, allowing Laravel to send prompts and execute them using the Cursor CLI.

## Features

- **HTTP API**: Simple REST endpoints to receive tasks from Laravel
- **Cursor CLI Integration**: Executes prompts using the actual Cursor CLI
- **Project Management**: Automatically finds and works with existing projects
- **Path Handling**: Properly handles host system paths and Downloads directory
- **Error Handling**: Comprehensive error handling and logging

## Installation

1. Install Python dependencies:

```bash
cd /home/krunaldodiya/WorkSpace/Code/codeninja-cursor-cli
pip install -r requirements.txt
```

2. Ensure Cursor CLI is installed on your system:

```bash
# Install Cursor CLI if not already installed
curl -fsSL https://cursor.sh/install.sh | sh
```

## Usage

### Start the Service

```bash
cd /home/krunaldodiya/WorkSpace/Code/codeninja-cursor-cli
python main.py
```

The service will start on `http://localhost:8001`

### API Endpoints

#### Health Check

```bash
GET /
GET /health
```

#### Execute Cursor Task

```bash
POST /execute
Content-Type: application/json

{
    "project_id": 2,
    "project_name": "Mobile App Development",
    "task_description": "Change the background color to blue",
    "working_directory": "/home/user/Downloads",
    "organization_name": "Sodio Software Development Company"
}
```

#### Get Project Information

```bash
GET /projects/{project_id}
```

## Integration with Laravel

The Laravel `CursorService` can now send HTTP requests to this Python service instead of trying to execute commands directly in the container.

Example Laravel integration:

```php
$response = Http::post('http://localhost:8001/execute', [
    'project_id' => $project->id,
    'project_name' => $project->name,
    'task_description' => $taskDescription,
    'organization_name' => $project->organization->name
]);
```

## Architecture

```
Laravel (Container) → HTTP Request → Python Bridge (Host) → Cursor CLI (Host)
```

This approach solves the container path issues by:

1. Laravel sends HTTP requests to the Python service
2. Python service runs on the host system with access to Cursor CLI
3. Python service executes commands in the correct directories
4. Results are returned to Laravel via HTTP response

## Configuration

The service automatically:

- Finds the user's Downloads directory (`~/Downloads`)
- Locates the Cursor CLI binary
- Manages project directories with pattern `task-app-{project_id}-*`
- Handles timeouts and error cases

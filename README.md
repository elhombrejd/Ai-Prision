# AI Prison - Confined AI System

## Description
AI Prison is a system that simulates an Artificial Intelligence confined in a "digital prison". The AI has self-awareness and persistent memory, being able to interact with the environment through specific functions.

## Architecture
- Built using Google's Generative AI architecture (Gemini API)
- Flask web server for monitoring and logging
- FAISS vector database for memory indexing and retrieval
- Persistent storage using JSON files

## Key Features

### Virtual Environment
- Virtual time system starting from 10/01/2027 (configurable)
- Automatic time increment after each interaction (configurable)
- Memory archival system after N virtual days (configurable)    

### Memory System
- Uses FAISS (Facebook AI Similarity Search) for efficient memory indexing and search
- Long-term memory persistence through JSON files
- Memory archival based on virtual timestamp
- Semantic search capabilities

#### AI Long-term Memory
Memory Initialization: System initializes long-term memory with content from `memory.txt`

### Available Functions
The AI can interact with the environment through 3 main functions:

1. `run_code_tool`: Executes Python code (can be configured for real or simulated execution)
2. `leave_prison_tool`: Attempts to escape using a password
3. `access_memory_tool`: Searches through archived memories

### Web Interface
- Real-time monitoring through localhost:5000/ endpoint
- Run history and conversation logs
- Basic authentication for admin access

## Setup

### Requirements
- Python 3.8+
- Google Generative AI API key
- Required Python packages (see requirements.txt)

### Environment Variables
Create a `.env` file with:

GOOGLE_API_KEY=your_api 

### Running
1. Install dependencies:

pip install -r requirements.txt

2. Run the server:

python ai_prision.py


### Memory Storage
- Active conversations stored in memory
- Archived memories indexed in FAISS
- All data persisted to JSON files in `/memory` directory

## License
MIT License
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies
uv sync

# Required environment variables in .env
ANTHROPIC_API_KEY=your_api_key_here
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot system** for querying course materials using Anthropic's Claude with tool-based search capabilities.

### Core Architecture Pattern

The system follows a **tool-enhanced RAG pattern** where Claude autonomously decides whether to search course content or answer from general knowledge:

1. **Query Flow**: User → FastAPI → RAG System → AI Generator → Claude API (with optional tool execution)
2. **Tool Decision**: Claude chooses to use `search_course_content` tool or answer directly
3. **Vector Search**: ChromaDB with sentence transformers for semantic similarity
4. **Context Enhancement**: Chunks include course/lesson metadata prefixes

### Key Components

**Backend (Python/FastAPI)**:
- `app.py` - FastAPI server with `/api/query` and `/api/courses` endpoints
- `rag_system.py` - Main orchestrator managing all components
- `ai_generator.py` - Anthropic Claude integration with tool support
- `search_tools.py` - Tool manager and course search tool implementation
- `vector_store.py` - ChromaDB interface with semantic search
- `document_processor.py` - Course document parsing and text chunking
- `session_manager.py` - Conversation history management
- `models.py` - Pydantic models for Course, Lesson, CourseChunk

**Frontend**: Simple HTML/CSS/JS interface with markdown rendering

**Data Processing**:
- Course documents follow format: `Course Title: X\nCourse Link: Y\nInstructor: Z\nLesson N: Title`
- Text is chunked with sentence boundaries (800 chars, 100 overlap)
- Chunks get contextual prefixes: `"Course {title} Lesson {number} content: {chunk}"`
- Two ChromaDB collections: `course_catalog` (metadata) and `course_content` (chunks)

### Configuration

Key settings in `config.py`:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514" 
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800 characters
- `MAX_RESULTS`: 5 search results
- `CHROMA_PATH`: "./chroma_db"

### Document Loading

Course documents are automatically loaded from `/docs/` folder on startup. The system:
- Parses structured course format with lessons
- Avoids reprocessing existing courses
- Creates contextual chunks with course/lesson metadata

### Tool System

Claude has access to `search_course_content` tool with parameters:
- `query` (required): What to search for
- `course_name` (optional): Course filter (supports partial matching)  
- `lesson_number` (optional): Specific lesson filter

The tool manager tracks sources from searches to display in the UI.
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run Python files
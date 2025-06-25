# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Frontend (React + Vite)
```bash
cd frontend
npm install        # Install dependencies
npm run dev        # Start development server (http://localhost:5173)
npm run build      # Build for production
npm run lint       # Run ESLint
```

### Backend (LangGraph + FastAPI)
```bash
cd backend
pip install .      # Install package
langgraph dev      # Start development server (http://127.0.0.1:2024)
make test          # Run unit tests with pytest
make lint          # Run ruff linting and mypy type checking
make format        # Format code with ruff
```

### Full Stack Development
```bash
make dev           # Start both frontend and backend servers concurrently
```

## Architecture Overview

This is a fullstack research assistant application with the following key components:

### Backend (LangGraph Agent)
- **Core Graph**: `backend/src/agent/graph.py` - Main LangGraph workflow with nodes for query generation, web research, reflection, and answer finalization
- **State Management**: `backend/src/agent/state.py` - Defines typed state objects for different graph phases
- **Configuration**: `backend/src/agent/configuration.py` - Runtime configuration schema
- **FastAPI Server**: `backend/src/agent/app.py` - Serves the frontend and exposes LangGraph API

### Agent Workflow
1. **generate_query**: Uses Gemini 2.0 Flash to create optimized search queries
2. **web_research**: Executes parallel web searches using Google Search API with Gemini
3. **reflection**: Analyzes results to identify knowledge gaps and generate follow-up queries
4. **finalize_answer**: Synthesizes research into final answer with citations

### Frontend (React + TypeScript)
- **Main App**: `frontend/src/App.tsx` - Manages streaming connection to LangGraph backend
- **Components**: Built with Radix UI primitives and Tailwind CSS
- **Real-time Updates**: Uses `@langchain/langgraph-sdk/react` for streaming agent events

## Environment Setup

### Required Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (required for both development and production)
- Create `backend/.env` from `backend/.env.example`

### Development URLs
- Frontend: http://localhost:5173
- Backend API: http://127.0.0.1:2024
- LangGraph UI: Automatically opens when running `langgraph dev`

## Key Configuration Files
- `backend/langgraph.json`: LangGraph deployment configuration
- `backend/pyproject.toml`: Python dependencies and tool configuration (ruff, mypy)
- `frontend/package.json`: Node.js dependencies and scripts
- `pixi.toml`: Cross-platform package management (optional)

## Production Deployment
- Use `docker build -t gemini-fullstack-langgraph -f Dockerfile .` to build container
- Frontend is served at `/app` route by FastAPI server
- Requires Redis and Postgres for LangGraph Cloud deployment
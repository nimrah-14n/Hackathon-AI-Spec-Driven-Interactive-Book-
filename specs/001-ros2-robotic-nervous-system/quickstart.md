# Quickstart Guide: ROS 2 Educational Module - The Robotic Nervous System

**Created**: 2025-12-15
**Feature**: 001-ros2-robotic-nervous-system
**Status**: Complete

## Overview

This guide provides a quick start for developers to set up, run, and contribute to the ROS 2 Educational Module. This module is part of the AI/Spec-Driven Interactive Book with Integrated RAG Chatbot project.

## Prerequisites

- Python 3.11 or higher
- Node.js 18+ and npm/yarn
- PostgreSQL (for local development) or Neon Serverless Postgres account
- Qdrant Cloud account (or local Qdrant instance)
- OpenAI API key
- Better Auth account (or local auth setup)

## Setup Instructions

### 1. Clone and Initialize Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup (FastAPI)

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start the backend server:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup (Docusaurus)

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual API endpoints and keys
```

4. Start the development server:
```bash
npm run start
# or
yarn start
```

### 4. Database Setup

1. Set up Neon Serverless Postgres:
   - Create account at https://neon.tech
   - Create a new project
   - Update your `.env` file with connection string

2. For local development, you can use Docker:
```bash
docker run --name postgres-ros2-book -e POSTGRES_DB=ros2book -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:15
```

### 5. Vector Database Setup (Qdrant)

1. For Qdrant Cloud:
   - Create account at https://qdrant.tech
   - Create a new cluster
   - Update your `.env` file with cluster URL and API key

2. For local development:
```bash
docker run -d --name qdrant-vector -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 6. Content Initialization

1. Run the content indexing script to populate the vector database:
```bash
cd backend
source venv/bin/activate
python scripts/index_content.py
```

2. This will:
   - Load all chapter content from the docs directory
   - Generate embeddings using OpenAI
   - Store content chunks in Qdrant
   - Create database entries in Postgres

## Running the Application

### Development Mode

1. Start backend:
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

2. In a new terminal, start frontend:
```bash
cd frontend
npm run start
```

3. The application will be available at `http://localhost:3000`

### Production Mode

1. Build frontend:
```bash
cd frontend
npm run build
```

2. Serve frontend (Docusaurus builds to static files compatible with GitHub Pages)

3. Deploy backend to your preferred hosting platform

## Key Endpoints

### Backend (default: http://localhost:8000)
- `GET /modules/001-ros2-robotic-nervous-system/chapters` - List all chapters
- `GET /modules/001-ros2-robotic-nervous-system/chapters/{number}` - Get specific chapter
- `POST /rag/query` - RAG question answering
- `POST /subagents/ros-concept-explainer` - ROS concept explanations
- `POST /auth/register` - User registration

### Frontend (default: http://localhost:3000)
- `/` - Home page
- `/docs/module1-ros2` - ROS 2 module documentation
- `/docs/module1-ros2/{chapter-slug}` - Specific chapter

## Testing

### Backend Tests
```bash
cd backend
source venv/bin/activate
pytest
```

### Frontend Tests
```bash
cd frontend
npm run test
```

## Environment Variables

Required environment variables for the backend:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost/dbname

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Better Auth
BETTER_AUTH_SECRET=your_secret_key
BETTER_AUTH_URL=http://localhost:8000

# Application
API_BASE_URL=http://localhost:8000
FRONTEND_BASE_URL=http://localhost:3000
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   - Ensure PostgreSQL is running
   - Verify connection string in `.env`
   - Run migrations: `alembic upgrade head`

2. **Vector Database Issues**:
   - Ensure Qdrant is running
   - Check API key and URL in `.env`
   - Verify network connectivity

3. **Frontend Build Issues**:
   - Clear npm cache: `npm cache clean --force`
   - Delete node_modules and reinstall: `rm -rf node_modules && npm install`

4. **OpenAI API Issues**:
   - Verify API key is valid
   - Check rate limits
   - Ensure billing is set up on OpenAI platform

## Next Steps

1. Customize chapter content in the `docs/module1-ros2/` directory
2. Extend the RAG system with additional content sources
3. Add more specialized subagents for different ROS 2 concepts
4. Implement additional personalization features
5. Add Urdu translations for all chapters

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `pytest` (backend) and `npm run test` (frontend)
6. Commit your changes: `git commit -m 'Add new feature'`
7. Push to the branch: `git push origin feature/new-feature`
8. Submit a pull request
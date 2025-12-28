# UniGuide Backend API

RAG (Retrieval-Augmented Generation) system for institutional knowledge management, powered by Supabase and Google Gemini.

## Tech Stack

- **Framework:** FastAPI
- **Database:** Supabase (PostgreSQL + pgvector)
- **Auth:** Supabase Auth
- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
- **Deployment:** Render.com

## Quick Start

### Prerequisites

- Python 3.10+
- Supabase account with configured project
- Google Gemini API key

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/uniguide-backend.git
cd uniguide-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development, also install dev dependencies
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Running Locally

```bash
# Development server (with auto-reload)
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --port 8000
```

Server runs at: http://localhost:8000

### API Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Deployment

### Render.com

1. Connect this repository to Render
2. Create a new Web Service
3. Render will auto-detect the `render.yaml` configuration
4. Add environment variables in Render dashboard:
   - `GEMINI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`

### Docker

```bash
docker build -t uniguide-backend .
docker run -p 8000:8000 --env-file .env uniguide-backend
```

## API Endpoints

### Public

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/api/chat/query` | Student chat query |
| GET | `/api/circular/latest` | Get latest circular |
| POST | `/api/circular/chat` | Chat about circulars |

### Admin (Requires Auth)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/login` | Admin login |
| POST | `/api/admin/upload` | Upload document |
| GET | `/api/admin/documents` | List documents |
| GET | `/api/admin/stats` | Dashboard stats |
| DELETE | `/api/admin/documents/{id}` | Delete document |
| PUT | `/api/admin/documents/{id}/rename` | Rename document |
| PUT | `/api/admin/documents/{id}/metadata` | Update metadata |

## Development

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy . --ignore-missing-imports

# Format code
ruff format .
```

### Project Structure

```
uniguide-backend/
├── main.py                 # FastAPI application
├── config.py               # Settings management
├── models/
│   └── schemas.py          # Pydantic models
├── routes/
│   ├── admin.py            # Admin endpoints
│   ├── chat.py             # Chat endpoints
│   └── circular.py         # Circular endpoints
├── services/
│   ├── auth.py             # Supabase auth
│   ├── supabase_auth.py    # Auth dependencies
│   ├── supabase_client.py  # Supabase client
│   ├── vector_store.py     # Vector store ops
│   ├── rag_engine.py       # RAG implementation
│   └── document_processor.py
└── uploads/                # Document storage
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | Supabase anon/public key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `UPLOAD_DIRECTORY` | No | Upload path (default: ./uploads) |
| `MAX_FILE_SIZE_MB` | No | Max file size (default: 50) |

## License

MIT License

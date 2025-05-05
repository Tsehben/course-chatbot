# GENED 1188 Course Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about GENED 1188 course materials using OpenAI's language models.

## 🚀 Features

- **Course-Specific AI Assistant**: Provides answers grounded in GENED 1188 course materials
- **RAG Architecture**: Uses vector embeddings to retrieve relevant course content before generating responses
- **User Authentication**: Secure email/password authentication with JWT tokens
- **Modern Web Interface**: Responsive chat interface with session management
- **Document Processing**: Supports multiple document formats (PDF, DOCX, PPTX, TXT, MD)

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.x
- **Vector Database**: ChromaDB for document embeddings
- **AI Models**: OpenAI GPT-4o-mini for chat completion, text-embedding-3-small for embeddings
- **Authentication**: JWT-based auth with bcrypt password hashing
- **Frontend**: HTML/CSS/JavaScript with responsive design

## 📋 Prerequisites

- Python 3.8+ 
- OpenAI API key
- (Optional) JWT secret for production deployment

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd course-chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file with the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   JWT_SECRET=your_jwt_secret_for_auth
   ```

5. **Ingest course materials**
   - Place your course documents in the `materials/` directory
   - Run the ingestion script:
     ```bash
     python ingest.py
     ```

## 🚀 Running the Application

```bash
uvicorn app:app --reload
```

The application will be available at http://localhost:8000

## 🔄 API Endpoints

- **GET /** - Main chat interface
- **POST /ask** - Submit questions to the chatbot
- **POST /auth/signup** - Create a new user account
- **POST /auth/login** - Authenticate and receive JWT token

## 📝 Notes

- The vector database (ChromaDB) is stored locally in the `chroma/` directory
- User credentials are stored in `users.db` using SQLite
- For production, consider using a more robust database solution

## 🔒 Security Considerations

- The current authentication system is for demonstration purposes
- In production, implement proper password reset flows and account management
- Consider adding rate limiting to prevent API abuse

## 📚 Project Structure

```
course-chatbot/
├── app.py              # Main FastAPI application
├── auth.py             # Authentication system
├── ingest.py           # Document ingestion script
├── retrieval.py        # Vector search functionality
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not in repo)
├── materials/          # Course documents to be ingested
├── chroma/             # Vector database storage
├── front-end/          # Web interface assets
└── templates/          # HTML templates (fallback)
```

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    logger.info("🚀 Starting API...")
    
    try:
        rag_pipeline = RAGPipeline(
            model_name=os.getenv("LLM_MODEL", "mistral"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chroma_db_dir=os.getenv("CHROMA_DB_DIR", "./chroma_db"),
            kb_dir=os.getenv("KB_DIR", "./knowledge_base")
        )
        logger.info("✅ RAG Pipeline ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    yield
    logger.info("🛑 Shutting down...")

app = FastAPI(
    title="🤖 Open-Source RAG API",
    description="Free RAG powered by Ollama & Chroma",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=10)

class SourceDocument(BaseModel):
    source: str
    row: int
    content: str
    similarity_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query_time_ms: float
    model: str
    status: str = "success"

@app.get("/health")
async def health_check():
    """Check API health"""
    return {
        "status": "healthy" if rag_pipeline else "degraded",
        "api_version": "1.0.0",
        "llm_model": rag_pipeline.model_name if rag_pipeline else "N/A",
        "kb_loaded": rag_pipeline.vectorstore is not None if rag_pipeline else False
    }

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Open-Source RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/upload")
async def upload_knowledge_base(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload Excel file"""
    try:
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only .xlsx and .xls files")
        
        logger.info(f"📤 Uploading: {file.filename}")
        
        Path(rag_pipeline.kb_dir).mkdir(exist_ok=True)
        
        file_path = os.path.join(rag_pipeline.kb_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        kb_name = file.filename.split(".")[0]
        background_tasks.add_task(_process_kb_async, file_path, kb_name)
        
        logger.info(f"📚 Processing: {kb_name}")
        
        return {
            "status": "processing",
            "filename": file.filename,
            "kb_name": kb_name,
            "message": "Processing started..."
        }
    
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_kb_async(file_path: str, kb_name: str):
    """Background processing"""
    try:
        logger.info(f"🔄 Processing KB: {kb_name}")
        success = rag_pipeline.ingest_excel(file_path, kb_name)
        if success:
            logger.info(f"✅ KB ready: {kb_name}")
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")

@app.get("/list-kb")
async def list_knowledge_bases():
    """List all KBs"""
    try:
        kb_dir = Path(rag_pipeline.kb_dir)
        if not kb_dir.exists():
            return {"count": 0, "knowledge_bases": []}
        
        import pandas
        kbs = []
        for csv_file in kb_dir.glob("*.csv"):
            df = pandas.read_csv(csv_file)
            kbs.append({
                "name": csv_file.stem,
                "size": csv_file.stat().st_size,
                "document_count": len(df)
            })
        
        return {"count": len(kbs), "knowledge_bases": kbs}
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-kb")
async def load_knowledge_base(kb_name: str):
    """Load KB"""
    try:
        logger.info(f"📂 Loading: {kb_name}")
        success = rag_pipeline.load_existing_kb(
            collection_name=kb_name.replace(" ", "_").lower()
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Loaded: {kb_name}",
                "kb_name": kb_name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to load")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query KB"""
    try:
        if not rag_pipeline.qa_chain:
            raise HTTPException(status_code=400, detail="Load KB first")
        
        import time
        start_time = time.time()
        
        result = rag_pipeline.query(request.question, top_k=request.top_k)
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        sources = [
            SourceDocument(
                source=s.get("source"),
                row=s.get("row"),
                content=s.get("content"),
                similarity_score=s.get("similarity_score")
            )
            for s in result.get("sources", [])
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            query_time_ms=elapsed_ms,
            model=rag_pipeline.model_name,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )

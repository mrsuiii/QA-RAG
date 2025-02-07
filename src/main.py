from fastapi import FastAPI, Request, Response,UploadFile,HTTPException, Depends
from pydantic import BaseModel
from rag.ingestion import DocumentIngestor
from rag.retrivial import Retrivial
from fastapi.security import APIKeyHeader
from config import API_KEY
import logging,time
import uvicorn

metrics = {
    "total_requests": 0,
    "total_success_requests": 0,
    "total_failed_requests": 0,
    "total_process_time": 0.0,
    "ingest_requests": 0,
    "generate_requests": 0,
    "total_tokens_used": 0,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME,auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """A dependency to verify the API key in the request header."""
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

app = FastAPI(
    title = "RAG QA",
    description=(
        "A Retrieval-Augmented Generation system for technical documentation. "
        "Endpoints include document ingestion and question answering with context. "
        "Monitoring is implemented for token usage, response times, and success/failure rates."
    ),
    dependencies=[Depends(verify_api_key)]
)


class input_prompts(BaseModel):
    query: str 

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """A middleware to track API metrics for each request."""
    global metrics
    metrics["total_requests"] += 1
    start_time = time.time()

    response: Response = await call_next(request)
    process_time = time.time() - start_time
    metrics["total_process_time"] += process_time

    # Update success/failure counts based on HTTP status.
    if response.status_code < 400:
        metrics["total_success_requests"] += 1
    else:
        metrics["total_failed_requests"] += 1

    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response



@app.post("/ingest", summary = "Upload and ingest a document file (PDF or Markdown).",tags=["Document Ingestion"])
async def ingest(document: UploadFile):
    """Ingest a document file (PDF or Markdown) by uploading it to the API."""
    # Validate filename
    filename = document.filename
    global metrics
    metrics["ingest_requests"] += 1
    if not filename or '.' not in filename:
        raise HTTPException(status_code=400, detail="Invalid or missing filename.")
    
    # Extract and normalize the file extension.
    extension = filename.split('.')[-1].lower()

    # Validate based on file extension and MIME type.
    if extension == "pdf":
        if document.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid MIME type for PDF file.")
    elif extension == "md":
        if document.content_type not in ["text/markdown", "text/plain"]:
            raise HTTPException(status_code=400, detail="Invalid MIME type for Markdown file.")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Only PDF and Markdown files are allowed."
        )

    try:
        # Read file content once.
        file_content = await document.read()
       
        doc_ingestor = DocumentIngestor()
        doc_ingestor.run_from_api(file_content, filename, extension)
        logger.info(f"File '{filename}' ingested successfully.")
    except Exception as e:
        logger.error(f"Error processing file '{filename}': {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process document ingestion.")

    return {"detail": "File processed successfully."}

@app.post("/generate", summary = "Answer a question with context",tags=["Question Answering"])
async def generate(input_prompts: input_prompts):
    """
    Generate a response to a question based on the provided context."""
    query = input_prompts.query
    global metrics
    metrics["generate_requests"] += 1
    try: 
        retrivier = Retrivial(query)
        response_text,formatted_context,token_len = retrivier.run()
        metrics["total_tokens_used"] += token_len
        logger.info("Query Processed Successfully")
    except Exception as e:
        logger.error(f"Error in processing query: {str(e)}")
        return {"response": "Error in processing query"}

    return {"query": query,"response": response_text,"sources(context)":formatted_context ,"token_count": token_len}

@app.get("/stats", summary = "Get system metrics",tags=["Monitoring"])
async def get_metrics():
    """
    Retrieve API metrics including total requests, success/failure counts,
    average response time, request counts per endpoint, and token usage.
    """
    avg_process_time = (metrics["total_process_time"] / metrics["total_requests"]
                        if metrics["total_requests"] > 0 else 0.0)
    return {
        "total_requests": metrics["total_requests"],
        "total_success_requests": metrics["total_success_requests"],
        "total_failed_requests": metrics["total_failed_requests"],
        "average_process_time": round(avg_process_time, 4),
        "ingest_requests": metrics["ingest_requests"],
        "generate_requests": metrics["generate_requests"],
        "total_tokens_used": metrics["total_tokens_used"],
    }
@app.delete("/clear_database", summary = "Clear the Vectore Database",tags=["Document Ingestion"])
async def delete_database():
    """
    Clear all vectors/documents from the vector store.
    
    Returns:
        A JSON response with the number of vectors deleted.
    """
    try:
        doc_ingestor = DocumentIngestor()
        num_deleted = doc_ingestor.clear_database()
        logger.info(f"Deleted {num_deleted} vectors from the vector store.")
        return {"detail": f"Deleted {num_deleted} vectors from the database."}
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear the database.")

if __name__ == "__main__":
    uvicorn.run("main:app",host="0.0.0.0", port =8001, reload = True)
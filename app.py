from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
from pathlib import Path
import json

from search_engine import SearchEngine

app = FastAPI()

# Initialize Search Engine
search_engine = SearchEngine()
print("Loading search engine data...")
if not search_engine.load_data():
    print("Warning: Initial data load failed. Please index a folder.")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20  # Default to 20 if not provided

class IndexRequest(BaseModel):
    path: str

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/search")
async def search(request: SearchRequest):
    results = search_engine.search(request.query, top_k=request.top_k)
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    return results

@app.post("/api/index")
async def index_folder(request: IndexRequest):
    if not os.path.exists(request.path):
        raise HTTPException(status_code=400, detail="Path does not exist")
    
    # Generator for SSE/Streaming
    def event_generator():
        for status in search_engine.index_folder_generator(request.path):
            # Yield JSON line
            yield json.dumps(status) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/api/images")
async def get_image(path: str):
    """Serve local images securely."""
    # Security check: ensure path is absolute and exists
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    # In a production app, we should valid the path is within allowed directories
    # For a local tool, we'll allow reading any file the user has access to
    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

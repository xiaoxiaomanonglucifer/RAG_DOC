"""
FastAPI endpoints
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import logging
from tools.openai_compat import router as openai_router

from config.settings import settings
from work.vector_store import vector_store
from work.retrieval import retriever
from work.models import model_manager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models, vector store, and retriever on startup"""
    logger.info("🚀 Starting RAG API server...")

    # Initialize models (embeddings, LLM, reranker)
    model_manager.initialize()

    # Initialize vector store
    vector_store.initialize()

    # Initialize retriever
    retriever.initialize()

    logger.info(f"✅ Server ready with {vector_store.get_count()} chunks indexed")
    yield
    # Cleanup on shutdown (if needed)
    logger.info("👋 Shutting down RAG API server...")

app = FastAPI(
    title="RAG API",
    description="Hybrid semantic + keyword search with reranking",
    version="2.0",
    lifespan=lifespan
)

# CORS support for n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for n8n
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)

# Serve PDF files for source linking
@app.get("/pdf/{filename}")
async def get_pdf(filename: str):
    """Serve PDF file for viewing sources"""
    pdf_path = Path(settings.PROCESSED_DIR) / filename
    if not pdf_path.exists():
        # Try upload dir as fallback
        pdf_path = Path(settings.UPLOAD_DIR) / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename
    )

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "RAG API",
        "version": "2.0",
        "supported_formats": [
            "PDF (.pdf)",
            "Word (.docx, .doc)",
            "Excel (.xlsx, .xls)"
        ],
        "features": [
            "Hybrid search (semantic + keyword)",
            "Cross-encoder reranking",
            "Multilingual embeddings",
            "Multi-format document support"
        ],
        "endpoints": {
            "/ingest": "POST - Upload document (PDF/Word/Excel)",
            "/query": "POST - Query knowledge base",
            "/documents": "GET - List all documents",
            "/documents/{filename}": "DELETE - Delete specific document",
            "/status": "GET - System status"
        }
    }

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    password: str = Form(None),
    knowledge_base: str = Form("default"),
):
    """Upload and process document (PDF, Word, Excel)

    Args:
        file: Document file to upload
        password: Optional password for encrypted PDFs (use Form field)
    """

    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls']
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    logger.info("=" * 60)
    logger.info(f"📄 NEW FILE: {file.filename} ({file_extension})")
    logger.info(f"🔑 Password received in API: {'YES' if password else 'NO'} (value: {repr(password)})")
    if password:
        logger.info("🔒 Password-protected file")
    logger.info("=" * 60)

    temp_filepath = Path(settings.UPLOAD_DIR) / file.filename

    try:
        # Save file
        logger.info("⏳ Saving file...")
        with open(temp_filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process document (supports PDF, DOCX, XLSX)
        kb = knowledge_base or "default"
        result = vector_store.ingest_document(
            str(temp_filepath),
            file.filename,
            password=password,
            knowledge_base=kb,
        )

        # Move to processed
        processed_path = Path(settings.PROCESSED_DIR) / file.filename
        shutil.move(temp_filepath, processed_path)
        logger.info(f"📁 Moved to: {processed_path}")

        # Rebuild BM25 with new docs（仅默认知识库重建混合检索索引）
        if kb == "default":
            retriever.rebuild_bm25()

        logger.info("=" * 60)

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            **result
        })

    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        if temp_filepath.exists():
            temp_filepath.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(
    question: str = Body(...),
    use_reranking: bool = Body(True),
    include_context: bool = Body(False),
    knowledge_base: str = Body("default"),
):
    """Query the RAG system with detailed response"""

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"🔍 Query: {question}")

    try:
        kb = knowledge_base or "default"
        result = retriever.query(question, use_reranking, knowledge_base=kb)

        # Extract detailed sources
        sources = []
        for i, doc in enumerate(result['source_documents'][:5], 1):
            source_info = {
                "rank": i,
                "filename": doc.metadata.get('filename', 'Unknown'),
                "page": doc.metadata.get('page', '?'),
                "preview": doc.page_content[:200].replace('\n', ' ') + "..."
            }
            if include_context:
                source_info["full_content"] = doc.page_content
            sources.append(source_info)

        logger.info("✓ Answer generated")

        response = {
            "question": question,
            "answer": result['result'],
            "sources": sources,
            "metadata": {
                "reranked": result.get('reranked', False),
                "num_chunks_used": result.get('num_chunks_used', len(result['source_documents'])),
                "top_relevance_score": float(result.get('top_score', 0))
            }
        }

        return response

    except Exception as e:
        logger.error(f"❌ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status(knowledge_base: str = "default"):
    """System status"""
    return {
        "status": "online",
        "chunks_indexed": vector_store.get_count(knowledge_base=knowledge_base or "default"),
        "model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "features": {
            "hybrid_search": True,
            "reranking": settings.USE_RERANKING,
            "multilingual": True
        }
    }

@app.get("/documents")
async def list_documents(knowledge_base: str = "default"):
    """List all documents in knowledge base"""
    
    try:
        kb = knowledge_base or "default"
        collection = vector_store.get_store_for_knowledge_base(kb)._collection
        results = collection.get()
        
        # Get unique filenames
        filenames = set()
        filename_stats = {}
        
        for metadata in results['metadatas']:
            if metadata and 'filename' in metadata:
                fname = metadata['filename']
                filenames.add(fname)
                
                # Count chunks per file
                if fname not in filename_stats:
                    filename_stats[fname] = {
                        'filename': fname,
                        'chunks': 0,
                        'upload_date': metadata.get('upload_date', 'Unknown')
                    }
                filename_stats[fname]['chunks'] += 1
        
        documents = list(filename_stats.values())
        documents.sort(key=lambda x: x['upload_date'], reverse=True)
        
        return {
            "total_documents": len(documents),
            "total_chunks": vector_store.get_count(knowledge_base=kb),
            "documents": documents,
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str, knowledge_base: str = "default"):
    """
    Delete a specific document from knowledge base
    
    Args:
        filename: Name of file to delete (e.g., "document.pdf")
    """
    
    try:
        kb = knowledge_base or "default"
        logger.info(f"🗑️  Deleting document: {filename} (knowledge_base={kb})")
        
        # Get collection
        collection = vector_store.get_store_for_knowledge_base(kb)._collection
        
        # Count chunks before deletion
        all_results = collection.get()
        chunks_to_delete = sum(
            1 for meta in all_results['metadatas'] 
            if meta and meta.get('filename') == filename
        )
        
        if chunks_to_delete == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Document '{filename}' not found in knowledge base"
            )
        
        # Delete by metadata filter
        collection.delete(
            where={"filename": filename}
        )
        
        logger.info(f"  ✓ Deleted {chunks_to_delete} chunks")
        
        # Rebuild BM25 index with remaining documents（仅默认知识库）
        if kb == "default":
            logger.info("  🔧 Rebuilding search index...")
            retriever.rebuild_bm25()
        
        new_total = vector_store.get_count(knowledge_base=kb)
        logger.info(f"  ✓ Remaining chunks: {new_total}")
        
        return {
            "status": "success",
            "deleted_document": filename,
            "chunks_deleted": chunks_to_delete,
            "remaining_chunks": new_total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def delete_all_documents(confirm: bool = False, knowledge_base: str = "default"):
    """
    Delete ALL documents from knowledge base
    
    Args:
        confirm: Must be true to confirm deletion (safety check)
    """
    
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to delete all documents"
        )
    
    try:
        kb = knowledge_base or "default"
        logger.info(f"🗑️  DELETING ALL DOCUMENTS (knowledge_base={kb})")
        
        chunks_before = vector_store.get_count(knowledge_base=kb)
        
        # Delete all from collection
        collection = vector_store.get_store_for_knowledge_base(kb)._collection
        
        # Get all IDs
        all_results = collection.get()
        if all_results['ids']:
            collection.delete(ids=all_results['ids'])
        
        logger.info(f"  ✓ Deleted {chunks_before} chunks")
        
        # Rebuild empty BM25（仅默认知识库）
        if kb == "default":
            retriever.rebuild_bm25()
        
        logger.info("  ✓ Knowledge base cleared")
        
        return {
            "status": "success",
            "message": "All documents deleted",
            "chunks_deleted": chunks_before,
            "remaining_chunks": 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error deleting all documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy"}

@app.get("/inspect_chunks")
async def inspect_chunks(
    filename: str = None,
    search_text: str = None,
    limit: int = 10,
    knowledge_base: str = "default",
):
    """Inspect chunks dalam database"""
    
    kb = knowledge_base or "default"
    collection = vector_store.get_store_for_knowledge_base(kb)._collection
    results = collection.get()
    
    chunks = []
    for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        # Filter by filename if specified
        if filename and metadata.get('filename') != filename:
            continue
        
        # Filter by search text if specified
        if search_text and search_text.lower() not in doc_text.lower():
            continue
            
        chunks.append({
            "index": i,
            "filename": metadata.get('filename'),
            "page": metadata.get('page'),
            "length": len(doc_text),
            "preview": doc_text[:500],
            "full_content": doc_text
        })
        
        if len(chunks) >= limit:
            break
    
    return {
        "total_in_db": len(results['documents']),
        "returned": len(chunks),
        "chunks": chunks
    }


@app.get("/knowledge_bases")
async def list_knowledge_bases():
    """
    List all available knowledge bases.

    - 'default' 对应旧版的单一知识库（collection_name = settings.COLLECTION_NAME）
    - 其他名称与底层 Chroma collection_name 一一对应
    """
    try:
        items = vector_store.list_knowledge_bases()
        return {"knowledge_bases": items}
    except Exception as e:
        logger.error(f"❌ Error listing knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge_bases")
async def create_knowledge_base(name: str = Body(..., embed=True)):
    """
    Create a new (empty) knowledge base backed by its own Chroma collection.

    The name will be used directly as collection_name (except 'default',
    which is reserved and always maps to the legacy collection).
    """
    kb = (name or "").strip()
    if not kb:
        raise HTTPException(status_code=400, detail="Knowledge base name cannot be empty")
    if kb == "default":
        raise HTTPException(status_code=400, detail="'default' is a reserved knowledge base name")

    existing = [item["name"] for item in vector_store.list_knowledge_bases()]
    if kb in existing:
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge base '{kb}' already exists",
        )

    try:
        # 创建一个新的 collection；如果已存在会被上面的检查拦截
        vector_store.get_store_for_knowledge_base(kb)
        logger.info(f"📚 Created new knowledge base: {kb}")
        return {"name": kb}
    except Exception as e:
        logger.error(f"❌ Error creating knowledge base '{kb}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
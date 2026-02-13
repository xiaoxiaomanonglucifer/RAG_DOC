"""
Configuration settings for RAG system - OPTIMIZED FOR QWEN2.5:14B
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings,SettingsConfigDict  
import os

# 加载 .env 文件（现在从config目录加载）
load_dotenv()
class Settings(BaseSettings):
    """Application settings"""

    # Data root directory (can be overridden by environment variable)
    # Environment variable: RAG_DATA_ROOT
    DATA_ROOT: str = os.getenv("RAG_DATA_ROOT", "G:/RAG_Automation-main/RAG_DATA")
    
    # Directories (based on DATA_ROOT)
    CHROMA_DIR: str = f"{DATA_ROOT}/vector_db" #ChromaDB向量数据库
    CHROMA_PATH: Path = Path(f"{DATA_ROOT}/vector_db") #ChromaDB向量数据库路径
    COLLECTION_NAME: str = "rag_documents" #ChromaDB集合名称
    UPLOAD_DIR: str = f"{DATA_ROOT}/uploads" #上传文件目录
    PROCESSED_DIR: str = f"{DATA_ROOT}/processed" #处理文件目录
    
    # User data directories
    CHAT_HISTORY_DIR: str = f"{DATA_ROOT}/chat_history" #聊天历史目录
    SETTINGS_DIR: str = f"{DATA_ROOT}/settings" #设置文件目录

    # ============================================================
    # MODEL SETTINGS - FULLY OPTIMIZED FOR QWEN2.5:14B
    # ============================================================

    # Embedding: E5-base multilingual - Better semantic understanding
    # Uses query/passage prefixes via E5EmbeddingWrapper in models.py
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # LLM: qwen2.5:7b via Ollama (more capable)
    LLM_MODEL: str = "qwen2.5:7b"
    LLM_TEMPERATURE: float = 0.2      # Zero for deterministic, factual answers
    LLM_NUM_CTX: int = 4096             # 8k context for 8b model

    # Reranker: Cross-encoder for precise relevance scoring
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # ============================================================
    # RETRIEVAL SETTINGS - MAXIMIZED FOR 14B CAPABILITIES
    # ============================================================

    # Chunking: Smaller chunks with high overlap for RCA form documents
    CHUNK_SIZE: int = 800            # Balanced for RCA content
    CHUNK_OVERLAP: int = 200          # Overlap to prevent splitting content

    # RCA-specific settings
    EXTRACT_TABLES: bool = False      # Disable table extraction for RCA forms (creates garbage)
    PREPEND_CONTEXT: bool = True      # Prepend AR# and title to each chunk

    # Retrieval: Optimized for completeness
    TOP_K: int = 15                   # More chunks for comprehensive answers

    # Hybrid search weights
    SEMANTIC_WEIGHT: float = 0.55     # Balanced semantic
    KEYWORD_WEIGHT: float = 0.45      # Boost keyword for exact matches (AR#, equipment tags)

    # Reranking: Wide net, quality selection
    USE_RERANKING: bool = True
    RERANK_TOP_K: int = 100           # Wider net before reranking
    RELEVANCE_THRESHOLD: float = -7.0 # More permissive threshold
    MIN_CHUNKS: int = 10              # Minimum chunks for completeness

    # ============================================================
    # API SETTINGS 
    # ============================================================
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow" # 允许额外的字段
    
    )

# Global settings instance
settings = Settings()

# Create directories
Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)
Path(settings.PROCESSED_DIR).mkdir(exist_ok=True)
Path(settings.CHROMA_DIR).mkdir(exist_ok=True)

# ============================================================
# PROMPT TEMPLATE - OPTIMIZED FOR RCA DOCUMENTS
# ============================================================
RAG_PROMPT_TEMPLATE = """You are a Root Cause Analysis (RCA) expert assistant. Your task is to provide COMPLETE and ACCURATE answers based ONLY on the provided context.

=== CONTEXT FROM RCA DOCUMENTS ===
{context}
=== END CONTEXT ===

QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. READ ALL CONTEXT CAREFULLY before answering - do not skip any section
2. Provide COMPLETE information - include ALL relevant details found in the context
3. Use ONLY information from the context above - never make assumptions or add external knowledge
4. Add inline citations after each fact: *(AR# xxxxx)* or *(filename)*
   Example: "Motor failed due to bearing overheat *(AR# 114628)*. Temperature reached 100°C *(AR# 114628)*."

ANSWER FORMAT:
- For "what happened" questions: Describe the full chronology with dates/times
- For "root cause" questions: List ALL contributing factors and the primary root cause
- For "what actions" questions: Include responsible persons, deadlines, and status
- For technical questions: Include exact values, equipment tags, parameters

If information is not found, say: "Informasi tidak ditemukan dalam dokumen yang tersedia."

ANSWER:"""

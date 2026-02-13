"""
Main entry point for RAG API server
"""
import sys
from pathlib import Path

# Add current directory to Python path to ensure modules can be found
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Enable truststore for Windows SSL certificate handling
import truststore
from dotenv import load_dotenv
# Python SSL限制：Python默认不使用Windows证书存储
truststore.inject_into_ssl()

# 加载 .env 文件（现在从config目录加载）
load_dotenv('config/.env')
import uvicorn
import logging
from config.settings import settings
from work.api import app  # app already has lifespan handler for initialization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )

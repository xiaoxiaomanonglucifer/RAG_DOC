"""
Machine learning models initialization - OPTIMIZED FOR QWEN2.5:14B + E5 EMBEDDINGS
"""
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import List
import logging
import os
from config.settings import settings, RAG_PROMPT_TEMPLATE
logger = logging.getLogger(__name__)

"""
导入模型 √
"""

#e5嵌入模型使用这个包装器
class E5EmbeddingWrapper:
    """
    Custom wrapper for E5 embeddings that handles query/passage prefixing.
    E5 models require:
    - Queries prefixed with "query: "
    - Documents prefixed with "passage: "
    """
    def __init__(self, model_name: str):
        logger.info(f"🚀 开始加载嵌入模型: {model_name}")
        
        # 设置环境变量显示下载进度
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
        
        try:
            # 显示模型信息
            logger.info(f"🔍 检查模型是否存在...")
            
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True  # 允许加载自定义模型
            )
            self.is_e5_model = "e5" in model_name.lower()
            logger.info(f"  ✓ E5模型检测: {self.is_e5_model}")
            logger.info(f"  ✓ 模型加载完成")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            raise
            # e5嵌入模型必须要写的固定写法
            """
          e5模型训练时的特殊要求：
            文档嵌入：需要 "passage: " 前缀
            查询嵌入：需要 "query: " 前缀
            性能保证：前缀确保模型按预期方式工作
            "passage: " + document_text  # 文档嵌入
            "query: " + query_text      # 查询嵌入
            # 输入
texts = ["这是文档内容", "另一个文档"]

# 处理后
texts = ["passage: 这是文档内容", "passage: 另一个文档"]
            """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with 'passage: ' prefix for E5"""
        if self.is_e5_model:
            texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query with 'query: ' prefix for E5"""
        if self.is_e5_model:
            text = f"query: {text}"
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        return embedding.tolist()


class ModelManager:
    """Manages all ML models - optimized for qwen2.5:7b"""

    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.reranker = None
        self.prompt = None

    def initialize(self):
        """Initialize all models"""
        logger.info("="*60)
        logger.info("🚀 开始初始化所有模型...")
        logger.info("="*60)
        
        # Use custom E5 wrapper for better embedding performance
        logger.info(f"🧮 正在加载嵌入模型: {settings.EMBEDDING_MODEL}")

        if "e5" in settings.EMBEDDING_MODEL.lower():
            # Use custom E5 wrapper with proper prefixing
            self.embeddings = E5EmbeddingWrapper(settings.EMBEDDING_MODEL)
            logger.info("  ✓ 使用E5嵌入包装器（带查询/段落前缀）")
        else:
            # Standard HuggingFace embeddings
            logger.info("  📥 从HuggingFace下载/加载嵌入模型...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        logger.info("✓ 嵌入模型加载完成")

        logger.info(f"🤖 正在连接Ollama: {settings.LLM_MODEL}")
        logger.info(f"  上下文窗口: {settings.LLM_NUM_CTX} tokens")
        self.llm = Ollama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            num_ctx=settings.LLM_NUM_CTX
        )
        logger.info("✓ LLM连接完成")

        if settings.USE_RERANKING:
            logger.info(f"🎯 正在加载重排序模型: {settings.RERANKER_MODEL}")
            logger.info("  📥 从HuggingFace下载/加载重排序模型...")
            self.reranker = CrossEncoder(settings.RERANKER_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("✓ 重排序模型加载完成")

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        logger.info("="*60)
        logger.info("🎉 所有模型初始化完成！")
        logger.info("="*60)

#全局模型管理器
model_manager = ModelManager()

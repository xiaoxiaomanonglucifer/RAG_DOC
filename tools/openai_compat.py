"""
OpenAI兼容API包装器，用于Open WebUI集成
支持对话历史和通过Ollama API的流式输出
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncIterator
import time
import logging
import json
import httpx

from work.retrieval import retriever
from config.settings import settings
from work.models import model_manager
from urllib.parse import quote
import re

logger = logging.getLogger(__name__)

# PDF链接的基础URL
import os
SERVER_IP = os.environ.get("SERVER_IP", "10.10.22.98")
PDF_BASE_URL = f"http://{SERVER_IP}:8000/pdf"
OLLAMA_BASE_URL = "http://localhost:11434"

def build_source_mapping(docs) -> dict:
    """构建AR号和文件名到PDF URL的映射"""
    mapping = {}
    for doc in docs:
        filename = doc.metadata.get('filename', '')
        ar_number = doc.metadata.get('ar_number', '')

        if filename:
            encoded = quote(filename)
            url = f"{PDF_BASE_URL}/{encoded}"
            name_only = filename.replace('.pdf', '')

            # 如果元数据中没有AR号，尝试从文件名提取
            if not ar_number:
                # 数字文件名 = AR号 (例如 "114628.pdf" -> "114628")
                if name_only.isdigit():
                    ar_number = name_only
                # EPR/OPR格式 (例如 "EPR-F2421-2023-06-1.pdf")
                elif name_only.startswith(('EPR-', 'OPR-', 'AROL')):
                    ar_number = name_only
                # 从文件名提取数字 (例如 "319_Notes.pdf" -> "319")
                else:
                    import re
                    num_match = re.match(r'^(\d+)', name_only)
                    if num_match:
                        ar_number = num_match.group(1)

            # 通过AR号映射（多种格式）
            if ar_number:
                mapping[ar_number] = url
                mapping[f"AR# {ar_number}"] = url
                mapping[f"ar #{ar_number}"] = url
                mapping[f"ar {ar_number}"] = url

                # 添加文件名映射
                mapping[filename] = url
                mapping[name_only] = url

    return mapping

def convert_citations_to_links(text: str, source_mapping: dict) -> str:
    """将引用格式转换为可点击链接"""

    # 查找引用的模式: *(something)*
    pattern = r'\*\(([^)]+)\)\*'

    def replace_citation(match):
        citation = match.group(1).strip()

        # 尝试找到此引用的URL
        url = source_mapping.get(citation)

        if not url:
            # 尝试部分匹配
            for key, val in source_mapping.items():
                if key in citation or citation in key:
                    url = val
                    break

        if url:
            return f'*([{citation}]({url}))*'
        else:
            return match.group(0)  # 如果没有匹配，返回原始内容

    return re.sub(pattern, replace_citation, text)

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0
    # 可选：指定使用哪个知识库，不传则使用默认知识库
    knowledge_base: Optional[str] = "default"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

def build_context_aware_query(messages: List[Message]) -> str:
    """构建带对话上下文的查询"""
    
    user_messages = [msg for msg in messages if msg.role == "user"]
    if not user_messages:
        return ""
    
    current_question = user_messages[-1].content
    
    # 如果是第一条消息，不需要上下文
    if len(messages) <= 1:
        return current_question
    
    # 构建对话上下文
    conversation = []
    for msg in messages[:-1]:  # 除当前消息外的所有消息
        if msg.role == "user":
            conversation.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            # 简化上下文（前150个字符）
            brief = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            conversation.append(f"Assistant: {brief}")
    
    # 限制上下文为最近5轮对话（10条消息）
    if len(conversation) > 10:
        conversation = conversation[-10:]
    
    # 构建增强查询
    context_prompt = "\n".join(conversation)
    enhanced_query = f"""Previous conversation:
{context_prompt}

Current question: {current_question}

Based on the conversation above, answer the current question."""
    
    logger.info(f"  📝 来自{len(conversation)}条消息的上下文")
    return enhanced_query

async def stream_response(query: str, simple_query: str, knowledge_base: Optional[str]) -> AsyncIterator[str]:
    """
    通用流式响应 - 支持Ollama和OpenAI
    Token在生成时立即出现
    """
    
    # 检查使用哪种模型类型
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
    
    if use_openai:
        logger.info("🌐 使用OpenAI API进行流式输出")
        return stream_openai_response(query, simple_query, knowledge_base)
    else:
        logger.info("🚀 使用Ollama API进行流式输出")
        return stream_ollama_response(query, simple_query, knowledge_base)

async def stream_ollama_response(query: str, simple_query: str, knowledge_base: Optional[str]) -> AsyncIterator[str]:
    """使用Ollama REST API进行真正的流式输出"""

    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    try:
        # 步骤1：检索文档（RAG）
        logger.info(f"  🔍 正在检索: {simple_query[:80]}...")
        kb = knowledge_base or "default"
        result = retriever.query(simple_query, use_reranking=True, knowledge_base=kb)
        docs = result.get('source_documents', [])

        # 步骤2：构建上下文和提示词
        context = "\n\n".join([doc.page_content for doc in docs[:settings.TOP_K]])
        prompt_text = model_manager.prompt.format(context=context, question=query)

        # 构建引用链接的源映射
        source_mapping = build_source_mapping(docs)

        logger.info("  🌊 通过Ollama API进行真正的流式输出...")

        # 步骤3：直接从Ollama REST API流式输出
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt_text,
            "stream": True,
            "options": {"temperature": 0, "num_gpu": 99}
        }

        buffer = ""
        total_chars = 0

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", ollama_url, json=payload) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if not token:
                            continue

                        buffer += token
                        total_chars += len(token)

                        # 缓冲引用直到完整
                        if '*(' in buffer and ')*' not in buffer:
                            continue

                        # 处理并输出
                        if buffer:
                            processed = convert_citations_to_links(buffer, source_mapping)
                            chunk_data = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": settings.LLM_MODEL,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": processed},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            buffer = ""

                        # 检查是否完成
                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        continue

        # 刷新剩余缓冲区和最终块
        if buffer:
            processed = convert_citations_to_links(buffer, source_mapping)
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': settings.LLM_MODEL, 'choices': [{'index': 0, 'delta': {'content': processed}, 'finish_reason': None}]})}\n\n"

        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': settings.LLM_MODEL, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"  ✓ Ollama流式输出完成 ({total_chars} 个字符)")

    except Exception as e:
        logger.error(f"❌ Ollama流式输出错误: {str(e)}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"

async def stream_openai_response(query: str, simple_query: str, knowledge_base: Optional[str]) -> AsyncIterator[str]:
    """使用OpenAI API直接进行流式输出"""
    
    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    
    try:
        # 步骤1：检索文档（RAG）
        logger.info(f"  🔍 正在检索: {simple_query[:80]}...")
        kb = knowledge_base or "default"
        result = retriever.query(simple_query, use_reranking=True, knowledge_base=kb)
        docs = result.get('source_documents', [])

        # 步骤2：构建上下文和提示词
        context = "\n\n".join([doc.page_content for doc in docs[:settings.TOP_K]])
        prompt_text = model_manager.prompt.format(context=context, question=query)

        # 构建引用链接的源映射
        source_mapping = build_source_mapping(docs)

        logger.info("  🌊 通过OpenAI API进行流式输出...")

        # 步骤3：从OpenAI API流式输出
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("需要OPENAI_API_KEY环境变量")
        
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        openai_payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
                {"role": "user", "content": prompt_text}
            ],
            "stream": True,
            "temperature": 0
        }

        buffer = ""
        total_chars = 0

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", "https://api.openai.com/v1/chat/completions", 
                                   json=openai_payload, headers=headers) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    if line.strip() == "data: [DONE]":
                        break
                    
                    try:
                        data = json.loads(line[6:])
                        if "choices" not in data or not data["choices"]:
                            continue
                            
                        delta = data["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        
                        if not token:
                            continue

                        buffer += token
                        total_chars += len(token)

                        # 缓冲引用直到完整
                        if '*(' in buffer and ')*' not in buffer:
                            continue

                        # 处理并输出
                        if buffer:
                            processed = convert_citations_to_links(buffer, source_mapping)
                            chunk_data = {
                                "id": data.get("id", chat_id),
                                "object": "chat.completion.chunk",
                                "created": data.get("created", created),
                                "model": data.get("model", "gpt-4"),
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": processed},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            buffer = ""

                        # 检查是否完成
                        if data["choices"][0].get("finish_reason") == "stop":
                            break

                    except json.JSONDecodeError:
                        continue

        # 刷新剩余缓冲区和最终块
        if buffer:
            processed = convert_citations_to_links(buffer, source_mapping)
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'gpt-4', 'choices': [{'index': 0, 'delta': {'content': processed}, 'finish_reason': None}]})}\n\n"

        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'gpt-4', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"  ✓ OpenAI流式输出完成 ({total_chars} 个字符)")

    except Exception as e:
        logger.error(f"❌ OpenAI流式输出错误: {str(e)}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI兼容的聊天完成接口，支持流式输出"""
    
    try:
        # 验证
        if not request.messages:
            return {"error": "No messages provided"}
        
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return {"error": "No user message found"}
        
        # 构建查询
        query = build_context_aware_query(request.messages)
        current_q = user_messages[-1].content
        kb = request.knowledge_base or "default"
        
        logger.info(f"🔍 Query: {current_q[:100]}...")
        logger.info(f"  📊 Messages: {len(request.messages)} | Stream: {request.stream}")

        # 流式响应
        if request.stream:
            return StreamingResponse(
                stream_response(query, simple_query=current_q, knowledge_base=kb),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        # 非流式响应
        use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
        
        if use_openai:
            return await get_openai_response(query, current_q, kb, request.model)
        else:
            return await get_ollama_response(query, current_q, kb)
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return {"error": str(e)}

async def get_openai_response(query: str, current_q: str, kb: str, model: str = "gpt-4"):
    """从OpenAI获取非流式响应"""
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {"error": "OPENAI_API_KEY environment variable is required"}
    
    # 步骤1：检索文档（RAG）
    result = retriever.query(current_q, use_reranking=True, knowledge_base=kb)
    docs = result.get('source_documents', [])
    
    # 步骤2：构建上下文和提示词
    context = "\n\n".join([doc.page_content for doc in docs[:settings.TOP_K]])
    prompt_text = model_manager.prompt.format(context=context, question=query)
    
    # 步骤3：调用OpenAI API
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model or "gpt-4",
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。使用提供的上下文来回答问题。"},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post("https://api.openai.com/v1/chat/completions", 
                                   json=payload, headers=headers)
        
    if response.status_code != 200:
        return {"error": f"OpenAI API error: {response.status_code}"}
    
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    
    # 处理引用
    source_mapping = build_source_mapping(docs)
    processed = convert_citations_to_links(content, source_mapping)
    
    return ChatCompletionResponse(
        id=data.get("id", f"chatcmpl-{int(time.time())}"),
        created=data.get("created", int(time.time())),
        model=data.get("model", model or "gpt-4"),
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": processed},
            "finish_reason": data["choices"][0].get("finish_reason", "stop")
        }]
    ).model_dump()

async def get_ollama_response(query: str, current_q: str, kb: str):
    """从Ollama获取非流式响应"""
    
    result = retriever.query(query, use_reranking=True, knowledge_base=kb)
    docs = result.get('source_documents', [])
    source_mapping = build_source_mapping(docs)
    processed = convert_citations_to_links(result['result'], source_mapping)

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=settings.LLM_MODEL,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": processed},
            "finish_reason": "stop"
        }]
    ).model_dump()

@router.get("/v1/models")
async def list_models():
    """列出可用模型（OpenAI兼容）"""
    models = []
    
    # 检查使用哪种模型类型（基于配置或环境变量）
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
    
    if use_openai:
        # 使用OpenAI模型
        openai_models = [
            "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"
        ]
        
        for model_name in openai_models:
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            })
            
        logger.info("🌐 返回OpenAI模型列表")
        
    else:
        # 使用Ollama模型
        
        # 1. 添加配置中的主要Ollama模型
        models.append({
            "id": settings.LLM_MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local-ollama"
        })
        
        # 2. 尝试获取Ollama中的所有模型
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                ollama_models = data.get("models", [])
                
                # 添加Ollama中的其他模型（避免重复）
                for model in ollama_models:
                    if model["name"] != settings.LLM_MODEL:  # 避免重复添加主要模型
                        models.append({
                            "id": model["name"],
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local-ollama"
                        })
                        
        except Exception as e:
            logger.warning(f"⚠️ 无法获取Ollama模型列表: {str(e)}")
            
        logger.info("🚀 返回Ollama模型列表")
    
    return {
        "object": "list",
        "data": models
    }
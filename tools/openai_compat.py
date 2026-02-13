"""
OpenAI-compatible API wrapper for Open WebUI integration
With conversation history support + streaming via Ollama API
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

# Base URL for PDF links
import os
SERVER_IP = os.environ.get("SERVER_IP", "10.10.22.98")
PDF_BASE_URL = f"http://{SERVER_IP}:8000/pdf"
OLLAMA_BASE_URL = "http://localhost:11434"

def build_source_mapping(docs) -> dict:
    """Build mapping from AR numbers and filenames to PDF URLs"""
    mapping = {}
    for doc in docs:
        filename = doc.metadata.get('filename', '')
        ar_number = doc.metadata.get('ar_number', '')

        if filename:
            encoded = quote(filename)
            url = f"{PDF_BASE_URL}/{encoded}"
            name_only = filename.replace('.pdf', '')

            # Try to extract AR# from filename if not in metadata
            if not ar_number:
                # Numeric filename = AR number (e.g., "114628.pdf" -> "114628")
                if name_only.isdigit():
                    ar_number = name_only
                # EPR/OPR format (e.g., "EPR-F2421-2023-06-1.pdf")
                elif name_only.startswith(('EPR-', 'OPR-', 'AROL')):
                    ar_number = name_only
                # Extract number from filename like "319_Notes.pdf" -> "319"
                else:
                    import re
                    num_match = re.match(r'^(\d+)', name_only)
                    if num_match:
                        ar_number = num_match.group(1)

            # Map by AR number (multiple formats)
            if ar_number:
                mapping[ar_number] = url
                mapping[f"AR# {ar_number}"] = url
                mapping[f"AR#{ar_number}"] = url
                mapping[f"AR {ar_number}"] = url

            # Map by filename variations
            mapping[name_only] = url
            mapping[filename] = url

    return mapping

def convert_citations_to_links(text: str, source_mapping: dict) -> str:
    """Convert inline citations like *(AR# 114628)* to clickable links"""

    # Pattern to find citations: *(something)*
    pattern = r'\*\(([^)]+)\)\*'

    def replace_citation(match):
        citation = match.group(1).strip()

        # Try to find URL for this citation
        url = source_mapping.get(citation)

        if not url:
            # Try partial matching
            for key, val in source_mapping.items():
                if key in citation or citation in key:
                    url = val
                    break

        if url:
            return f'*([{citation}]({url}))*'
        else:
            return match.group(0)  # Return original if no match

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
    """Build query with conversation context"""
    
    user_messages = [msg for msg in messages if msg.role == "user"]
    if not user_messages:
        return ""
    
    current_question = user_messages[-1].content
    
    # If first message, no context needed
    if len(messages) <= 1:
        return current_question
    
    # Build conversation context
    conversation = []
    for msg in messages[:-1]:  # All except current
        if msg.role == "user":
            conversation.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            # Brief context (first 150 chars)
            brief = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            conversation.append(f"Assistant: {brief}")
    
    # Limit context to last 5 exchanges (10 messages)
    if len(conversation) > 10:
        conversation = conversation[-10:]
    
    # Build enhanced query
    context_prompt = "\n".join(conversation)
    enhanced_query = f"""Previous conversation:
{context_prompt}

Current question: {current_question}

Based on the conversation above, answer the current question."""
    
    logger.info(f"  📝 Context from {len(conversation)} messages")
    return enhanced_query

async def stream_ollama_response(query: str, simple_query: str, knowledge_base: Optional[str]) -> AsyncIterator[str]:
    """
    TRUE streaming using Ollama REST API directly
    Tokens appear immediately as they are generated
    """

    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    try:
        # Step 1: Retrieve documents (RAG)
        logger.info(f"  🔍 Retrieving for: {simple_query[:80]}...")
        kb = knowledge_base or "default"
        result = retriever.query(simple_query, use_reranking=True, knowledge_base=kb)
        docs = result.get('source_documents', [])

        # Step 2: Build context and prompt
        context = "\n\n".join([doc.page_content for doc in docs[:settings.TOP_K]])
        prompt_text = model_manager.prompt.format(context=context, question=query)

        # Build source mapping for citation links
        source_mapping = build_source_mapping(docs)

        logger.info("  🌊 TRUE streaming via Ollama API...")

        # Step 3: Stream from Ollama REST API directly
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt_text,
            "stream": True,
            "options": {
                "temperature": 0,
                "num_gpu": 99,
            }
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

                        # Buffer citations until complete
                        if '*(' in buffer and ')*' not in buffer:
                            continue

                        # Process and yield
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

                        # Check if done
                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        continue

        # Flush remaining buffer
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

        # Final chunk
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': settings.LLM_MODEL, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"  ✓ Stream complete ({total_chars} chars)")

    except Exception as e:
        logger.error(f"❌ Stream error: {str(e)}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion with streaming support"""
    
    try:
        if not request.messages:
            return {"error": "No messages provided"}
        
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return {"error": "No user message found"}
        
        # Build context-aware query
        query = build_context_aware_query(request.messages)
        
        current_q = user_messages[-1].content
        logger.info(f"🔍 Query: {current_q[:100]}...")
        logger.info(f"  📊 Messages: {len(request.messages)} | Stream: {request.stream}")
        
        kb = request.knowledge_base or "default"

        # STREAMING RESPONSE
        if request.stream:
            return StreamingResponse(
                stream_ollama_response(query, simple_query=current_q, knowledge_base=kb),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        # NON-STREAMING RESPONSE (original behavior)
        else:
            result = retriever.query(query, use_reranking=True, knowledge_base=kb)
            docs = result.get('source_documents', [])
            source_mapping = build_source_mapping(docs)
            processed = convert_citations_to_links(result['result'], source_mapping)

            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=settings.LLM_MODEL,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": processed
                    },
                    "finish_reason": "stop"
                }]
            )

            return response.model_dump()
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return {"error": str(e)}

@router.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)"""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.LLM_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }
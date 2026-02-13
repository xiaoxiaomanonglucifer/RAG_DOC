"""
Advanced retrieval with hybrid search and reranking
"""
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
import logging
import re

from config.settings import settings
from work.models import model_manager
from work.vector_store import vector_store

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Manages hybrid retrieval (semantic + keyword) with query optimization"""

    def __init__(self):
        self.vector_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.qa_chain = None

    def expand_query(self, question: str) -> list[str]:
        """Smart query expansion for comprehensive retrieval

        Handles:
        - AR# number queries: Extract number and search by content
        - Table/structured data queries: Boost actual data tables
        - Structural queries (categories/groups): Boost table of contents
        """
        queries = [question]
        question_lower = question.lower()

        # Detect AR# queries and expand with content-based search
        ar_patterns = [
            r'AR[#\s]*(\d+)',           # AR# 114628, AR 114628
            r'ar[#\s]*(\d+)',           # ar# 114628
            r'(EPR-F\d+-\d+-\d+-\d+)',  # EPR-F2421-2023-06-1
            r'(OPR-F\d+-\d+-\d+-\d+)',  # OPR format
        ]

        for pattern in ar_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                ar_num = match.group(1)
                # Add filename-based search
                queries.append(f"{ar_num}.pdf")
                # Add content search without AR# prefix
                clean_query = re.sub(r'AR[#\s]*\d+', '', question, flags=re.IGNORECASE).strip()
                if clean_query and len(clean_query) > 5:
                    queries.append(clean_query)
                break

        # Detect tabular/structured data queries
        table_data_keywords = [
            "variables", "parameters", "tags", "settings", "configuration",
            "specifications", "values", "properties", "attributes",
            "failures", "errors", "issues", "causes", "solutions", "steps",
            "procedures", "requirements", "recommendations",
            "root cause", "penyebab", "kronologi", "chronology"
        ]
        is_table_data_query = any(keyword in question_lower for keyword in table_data_keywords)

        # Detect listing questions
        listing_keywords = ["apa saja", "what are", "what is", "sebutkan", "list", "daftar", "how many", "jelaskan"]
        is_listing_query = any(keyword in question_lower for keyword in listing_keywords)

        if is_listing_query:
            if is_table_data_query:
                queries.append(f"Table {question}")
            else:
                queries.append(f"contents {question}")

        return queries
        
    def initialize(self):
        """Initialize hybrid retriever"""
        
        logger.info("🔧 Setting up hybrid retrieval...")
        
        # Vector retriever (semantic)
        self.vector_retriever = vector_store.get_retriever()
        
        # Build BM25 retriever (keyword)
        self.rebuild_bm25()
        
        # Create ensemble
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[settings.SEMANTIC_WEIGHT, settings.KEYWORD_WEIGHT]
            )
            logger.info(f"  ✓ Hybrid search (semantic {settings.SEMANTIC_WEIGHT} + keyword {settings.KEYWORD_WEIGHT})")
        else:
            self.ensemble_retriever = self.vector_retriever
            logger.info("  ✓ Vector search only (no docs yet for BM25)")
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model_manager.llm,
            retriever=self.ensemble_retriever,
            chain_type_kwargs={"prompt": model_manager.prompt},
            return_source_documents=True
        )
        
        logger.info("✓ Retrieval system ready")
        
    def rebuild_bm25(self):
        """Rebuild BM25 retriever with current documents (default knowledge base only)"""
        
        # 为保持兼容性，这里仍然只针对默认知识库构建 BM25
        documents = vector_store.get_all_documents()
        
        if len(documents) > 0:
            logger.info(f"  🔧 Building BM25 index with {len(documents)} docs...")
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = settings.TOP_K
            
            # Update ensemble if it exists
            if self.ensemble_retriever and isinstance(self.ensemble_retriever, EnsembleRetriever):
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[settings.SEMANTIC_WEIGHT, settings.KEYWORD_WEIGHT]
                )
                # Update qa_chain retriever
                self.qa_chain.retriever = self.ensemble_retriever
            
            logger.info("  ✓ BM25 index updated")
        else:
            self.bm25_retriever = None
    
    def query(self, question: str, use_reranking: bool = None, knowledge_base: str | None = None) -> dict:
        """Query with OPTIMIZED reranking and smart relevance filtering"""

        use_reranking = use_reranking if use_reranking is not None else settings.USE_RERANKING

        # 规范化知识库名称（None 或 'default' 都视为默认）
        kb = knowledge_base or "default"

        if use_reranking and model_manager.reranker:
            # STEP 1: Multi-query retrieval for better coverage
            query_variations = self.expand_query(question)
            logger.info(f"  🔍 Using {len(query_variations)} query variations:")
            for i, qv in enumerate(query_variations, 1):
                logger.info(f"    {i}. {qv}")

            # 根据知识库获取对应的向量检索器
            retriever = vector_store.get_retriever(k=settings.RERANK_TOP_K, knowledge_base=kb)

            # Retrieve docs for each query variation and merge
            all_docs = []
            seen_contents = set()

            for query in query_variations:
                docs = retriever.invoke(query)
                for doc in docs:
                    # Deduplicate by content hash
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_docs.append(doc)

            retrieved_docs = all_docs[:settings.RERANK_TOP_K]  # Limit to RERANK_TOP_K
            logger.info(f"  📥 Retrieved {len(retrieved_docs)} unique candidate chunks")

            # STEP 2: Rerank with scores
            pairs = [[question, doc.page_content] for doc in retrieved_docs]
            scores = model_manager.reranker.predict(pairs)

            # Sort by relevance score
            doc_score_pairs = list(zip(retrieved_docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"  🎯 Reranking scores - Top 5:")
            for i, (doc, score) in enumerate(doc_score_pairs[:5], 1):
                preview = doc.page_content[:100].replace('\n', ' ')
                logger.info(f"    [{i}] Score: {score:.3f} | {preview}...")

            # STEP 3: STRICT filtering - quality over quantity
            filtered_docs = []
            for doc, score in doc_score_pairs:
                # STRICT: Only keep if score is decent
                if score >= settings.RELEVANCE_THRESHOLD:
                    filtered_docs.append(doc)
                    if len(filtered_docs) >= settings.TOP_K:
                        break

            # Fallback: If nothing passes threshold, take top MIN_CHUNKS
            if len(filtered_docs) == 0:
                logger.warning(f"  ⚠️ No chunks above threshold {settings.RELEVANCE_THRESHOLD}, taking top {settings.MIN_CHUNKS}")
                filtered_docs = [doc for doc, _ in doc_score_pairs[:settings.MIN_CHUNKS]]

            # Log score distribution
            avg_score = sum(score for _, score in doc_score_pairs[:len(filtered_docs)]) / len(filtered_docs) if filtered_docs else 0
            logger.info(f"  ✓ Selected {len(filtered_docs)} chunks (avg score: {avg_score:.2f})")

            # STEP 4: Build rich context with clear source markers
            context_parts = []
            for i, doc in enumerate(filtered_docs, 1):
                source = doc.metadata.get('filename', 'Unknown')
                ar_num = doc.metadata.get('ar_number', '')
                page = doc.metadata.get('page', '')

                # Clear header for easy LLM parsing
                if ar_num:
                    header = f"[Document {i}: AR# {ar_num} - {source}]"
                else:
                    header = f"[Document {i}: {source}]"
                if page:
                    header += f" (Page {page})"

                context_parts.append(f"{header}\n{doc.page_content}")

            context = "\n\n---\n\n".join(context_parts)

            # STEP 5: Query LLM with rich context
            prompt_text = model_manager.prompt.format(context=context, question=question)
            answer = model_manager.llm.invoke(prompt_text)

            return {
                'result': answer,
                'source_documents': filtered_docs,
                'reranked': True,
                'num_chunks_used': len(filtered_docs),
                'top_score': doc_score_pairs[0][1] if doc_score_pairs else 0
            }
        else:
            # Standard retrieval
            # 默认知识库仍然复用已有的 qa_chain，其他知识库按需临时创建链路
            if kb == "default" and self.qa_chain is not None:
                result = self.qa_chain({"query": question})
            else:
                tmp_retriever = vector_store.get_retriever(knowledge_base=kb)
                tmp_chain = RetrievalQA.from_chain_type(
                    llm=model_manager.llm,
                    retriever=tmp_retriever,
                    chain_type_kwargs={"prompt": model_manager.prompt},
                    return_source_documents=True
                )
                result = tmp_chain({"query": question})

            result['reranked'] = False
            return result
        
# Di retrieval.py, tambah method ini 

    def query_with_debug(self, question: str, use_reranking: bool = None, knowledge_base: str | None = None) -> dict:
        """Query dengan full debug info"""
        
        use_reranking = use_reranking if use_reranking is not None else settings.USE_RERANKING
        
        # Get raw retrieval results (before reranking)
        kb = knowledge_base or "default"
        retriever = vector_store.get_retriever(k=settings.RERANK_TOP_K, knowledge_base=kb)
        raw_docs = retriever.invoke(question)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {question}")
        logger.info(f"{'='*60}")
        
        # Log RAW retrieval (sebelum rerank)
        logger.info(f"\n📥 RAW RETRIEVAL ({len(raw_docs)} chunks):")
        for i, doc in enumerate(raw_docs[:10], 1):
            preview = doc.page_content[:200].replace('\n', ' ')
            source = doc.metadata.get('filename', 'unknown')
            logger.info(f"  [{i}] {source}: {preview}...")
        
        # Reranking
        if use_reranking and model_manager.reranker:
            pairs = [[question, doc.page_content] for doc in raw_docs]
            scores = model_manager.reranker.predict(pairs)
            
            # Log dengan scores
            logger.info(f"\n🎯 AFTER RERANKING (with scores):")
            doc_scores = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
            for i, (doc, score) in enumerate(doc_scores[:10], 1):
                preview = doc.page_content[:150].replace('\n', ' ')
                logger.info(f"  [{i}] Score: {score:.4f} | {preview}...")
            
            # Filter
            reranked_docs = [doc for doc, score in doc_scores[:settings.TOP_K]]
        else:
            reranked_docs = raw_docs[:settings.TOP_K]
        
        # Build context dan generate
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        prompt_text = model_manager.prompt.format(context=context, question=question)
        
        # Log final context yang dikirim ke LLM
        logger.info(f"\n📤 FINAL CONTEXT TO LLM ({len(context)} chars):")
        logger.info(f"{context[:1000]}...")
        
        answer = model_manager.llm.invoke(prompt_text)
        
        logger.info(f"\n💬 ANSWER: {answer[:500]}...")
        logger.info(f"{'='*60}\n")
        
        return {
            'result': answer,
            'source_documents': reranked_docs,
            'debug': {
                'raw_docs_count': len(raw_docs),
                'raw_docs': [(d.page_content[:200], d.metadata) for d in raw_docs[:5]],
                'final_docs': [(d.page_content[:200], d.metadata) for d in reranked_docs],
                'context_length': len(context)
            }
        }

# Global retriever
retriever = HybridRetriever()


"""
Vector store operations - Optimized for Academic Documents
"""
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from datetime import datetime
import logging
import re

from config.settings import settings
from work.models import model_manager

logger = logging.getLogger(__name__)

# ============================================================
# TEXT CLEANING PATTERNS FOR ACADEMIC DOCUMENTS
# ============================================================
NOISE_PATTERNS = [
    r'Page\s*\d+\s*of\s*\d+',
    r'©\s*\d{4}\s*.*?All\s*rights\s*reserved',
    r'DOI:\s*10\.\d+/.*',
    r'ISBN:\s*\d+-\d+-\d+-\d+-\d+',
    r'^\s*\d+\s*$',
    r'Abstract\s*$',
    r'Keywords?\s*$',
    r'References\s*$',
]

# Compile patterns for efficiency
COMPILED_NOISE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NOISE_PATTERNS]


def clean_academic_text(text: str) -> str:
    """Remove repetitive headers, footers, and noise from academic documents"""
    cleaned = text

    # Remove noise patterns
    for pattern in COMPILED_NOISE:
        cleaned = pattern.sub('', cleaned)

    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)

    # Remove lines that are just whitespace or very short
    lines = cleaned.split('\n')
    lines = [l for l in lines if len(l.strip()) > 2]
    cleaned = '\n'.join(lines)

    return cleaned.strip()


# ============================================================
# DOCUMENT METADATA EXTRACTION
# ============================================================
def extract_document_metadata(text: str, filename: str) -> dict:
    """Extract key metadata from academic document text"""
    metadata = {}
    
    # Extract title from common patterns
    title_patterns = [
        r'Title\s*[:\s]*([^\n]+)',
        r'title\s*[:\s]*([^\n]+)',
        r'^([^\n]{20,100})$',  # First line that might be title
    ]
    
    for pattern in title_patterns:
        title_match = re.search(pattern, text, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            title = re.sub(r'^[:\s]+', '', title)
            if len(title) > 10 and len(title) < 200:
                metadata['title'] = title
                break
    
    # Extract authors
    author_patterns = [
        r'Authors?\s*[:\s]*([^\n]+)',
        r'By\s*([^\n]+)',
        r'^([A-Z][a-z]+ [A-Z][a-z]+,.*[0-9]{4})',  # Academic citation format
    ]
    
    for pattern in author_patterns:
        author_match = re.search(pattern, text, re.IGNORECASE)
        if author_match:
            authors = author_match.group(1).strip()
            if len(authors) > 5:
                metadata['authors'] = authors[:200]
                break
    
    # Extract abstract
    abstract_match = re.search(r'Abstract\s*[:\n]*([^\n]*(?:\n[^\n]*){0,10})', text, re.IGNORECASE)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        if len(abstract) > 50:
            metadata['abstract'] = abstract[:500]
    
    # Extract keywords
    keywords_match = re.search(r'Keywords?\s*[:\n]*([^\n]*(?:\n[^\n]*){0,3})', text, re.IGNORECASE)
    if keywords_match:
        keywords = keywords_match.group(1).strip()
        if len(keywords) > 10:
            metadata['keywords'] = keywords[:200]
    
    # Extract DOI
    doi_match = re.search(r'DOI\s*[:\s]* (10\.\d+/[^\s]+)', text, re.IGNORECASE)
    if doi_match:
        metadata['doi'] = doi_match.group(1).strip()
    
    # Extract publication year
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        metadata['year'] = year_match.group(0)
    
    return metadata

class VectorStoreManager:
    """Manages ChromaDB vector store"""
    
    def __init__(self):
        self.vectorstore = None
        # 按 collection_name 缓存多个 Chroma 实例
        self._stores: dict[str, Chroma] = {}
        self.text_splitter = None
        
    def initialize(self):
        """Initialize vector store and text splitter"""

        # OPTIMIZED: Technical document-aware chunking
        # Larger chunks with smart boundaries to preserve context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n## ",            # Markdown headers (if present)
                "\n### ",           # Subheaders
                "\n\n\n",           # Triple newline (major sections)
                "\n\n",             # Double newline (paragraphs)
                "\nStep ",          # Procedure steps
                "\n\nStep ",        # Steps with spacing
                "\n• ",             # Bullet points
                "\n- ",             # Dashed lists
                "\n",               # Single newline
                ". ",               # Sentences
                " ",                # Words
            ],
            keep_separator=True,    # Keep separators for context
            length_function=len,
            is_separator_regex=False
        )

        # Initialize default ChromaDB vector store (默认知识库)
        logger.info("🗄️  Initializing ChromaDB vector store (default collection)...")
        default_store = Chroma(
            persist_directory=str(settings.CHROMA_PATH),
            embedding_function=model_manager.embeddings,
            collection_name=settings.COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        self.vectorstore = default_store
        self._stores[settings.COLLECTION_NAME] = default_store

        # 尝试发现已经存在的其他 collection（多知识库兼容）
        try:
            client = default_store._client  # type: ignore[attr-defined]
            existing = client.list_collections()
            for coll in existing:
                # chromadb Collection 对象通常有 .name 属性
                coll_name = getattr(coll, "name", None) or getattr(coll, "_name", None)
                if not coll_name or coll_name == settings.COLLECTION_NAME:
                    continue
                if coll_name in self._stores:
                    continue
                logger.info(f"🗄️  Found existing collection: {coll_name}")
                self._stores[coll_name] = Chroma(
                    persist_directory=str(settings.CHROMA_PATH),
                    embedding_function=model_manager.embeddings,
                    collection_name=coll_name,
                    collection_metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            logger.warning(f"⚠️ Unable to list existing collections: {e}")

        logger.info("✓ Vector store initialized")

    # ------------------------------------------------------------
    # 多知识库辅助方法
    # ------------------------------------------------------------
    def get_store_for_knowledge_base(self, knowledge_base: str | None = None) -> Chroma:
        """
        根据知识库名称获取/创建对应的 Chroma collection。
        - knowledge_base 为 None 或 'default' 时，映射到 settings.COLLECTION_NAME（旧逻辑）
        - 其他名称直接作为 collection_name 使用
        """
        if not knowledge_base or knowledge_base == "default":
            collection_name = settings.COLLECTION_NAME
        else:
            collection_name = knowledge_base

        if collection_name in self._stores:
            return self._stores[collection_name]

        logger.info(f"🗄️  Creating new collection for knowledge base: {collection_name}")
        store = Chroma(
            persist_directory=str(settings.CHROMA_PATH),
            embedding_function=model_manager.embeddings,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self._stores[collection_name] = store
        return store

    def list_knowledge_bases(self) -> list[dict]:
        """
        列出当前所有已存在的知识库名称。
        - 'default' 始终存在，对应 settings.COLLECTION_NAME
        - 其他 collection_name 原样返回
        """
        names = set()
        names.add("default")
        try:
            # 使用默认 store 的 client 列出 collection
            if self.vectorstore is not None:
                client = self.vectorstore._client  # type: ignore[attr-defined]
                for coll in client.list_collections():
                    coll_name = getattr(coll, "name", None) or getattr(coll, "_name", None)
                    if not coll_name or coll_name == settings.COLLECTION_NAME:
                        continue
                    names.add(coll_name)
        except Exception as e:
            logger.warning(f"⚠️ Unable to list knowledge bases: {e}")

        return [{"name": name} for name in sorted(names)]

    def get_count(self, knowledge_base: str | None = None) -> int:
        """Get number of chunks in database"""
        try:
            store = self.get_store_for_knowledge_base(knowledge_base)
            return store._collection.count()
        except Exception:
            return 0
    
    def get_all_documents(self, knowledge_base: str | None = None) -> list[Document]:
        """Retrieve all documents from vector store"""
        
        try:
            store = self.get_store_for_knowledge_base(knowledge_base)
            collection = store._collection
            results = collection.get()
            
            documents = []
            for i, doc_text in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                documents.append(Document(
                    page_content=doc_text,
                    metadata=metadata
                ))
            
            return documents
        except Exception as e:
            logger.warning(f"Could not load documents: {e}")
            return []
    
    def _is_valid_table(self, table: list, headers: list) -> bool:
        """Check if extracted table has meaningful structure"""
        if not table or len(table) < 2:
            return False

        # Check if headers are meaningful (not Col0, Col1, etc.)
        generic_headers = sum(1 for h in headers if re.match(r'^Col\d+$', str(h)))
        if generic_headers > len(headers) * 0.5:  # More than 50% generic
            return False

        # Check if table has reasonable dimensions
        if len(headers) > 20:  # Too many columns = likely form extraction garbage
            return False

        # Check if content is meaningful
        meaningful_cells = 0
        total_cells = 0
        for row in table[1:]:
            for cell in row:
                total_cells += 1
                if cell and len(str(cell).strip()) > 3:
                    meaningful_cells += 1

        if total_cells > 0 and meaningful_cells / total_cells < 0.3:  # Less than 30% meaningful
            return False

        return True

    def _extract_tables_from_pdf(self, filepath: str, filename: str) -> list[Document]:
        """Extract tables from PDF using pdfplumber - with quality filtering"""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("  ⚠️ pdfplumber not installed, skipping table extraction")
            return []

        table_chunks = []
        logger.info("  📊 Extracting tables with pdfplumber...")

        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()

                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 1:  # Has header + data
                                # Get headers
                                headers = table[0] if table[0] else [f"Col{i}" for i in range(len(table[1]))]
                                headers = [str(h).strip() if h else f"Col{i}" for i, h in enumerate(headers)]

                                # Skip low-quality tables
                                if not self._is_valid_table(table, headers):
                                    continue

                                # Build markdown table
                                md_lines = [
                                    f"## Table from {filename} - Page {page_num}, Table {table_idx + 1}",
                                    "",
                                    "| " + " | ".join(headers) + " |",
                                    "| " + " | ".join(["---"] * len(headers)) + " |"
                                ]

                                row_count = 0
                                for row in table[1:]:
                                    if row and any(cell for cell in row):
                                        cells = [str(cell).strip() if cell else "" for cell in row]
                                        # Pad or truncate to match headers
                                        while len(cells) < len(headers):
                                            cells.append("")
                                        cells = cells[:len(headers)]
                                        md_lines.append("| " + " | ".join(cells) + " |")
                                        row_count += 1

                                if row_count > 0:
                                    table_content = "\n".join(md_lines)
                                    chunk = Document(
                                        page_content=table_content,
                                        metadata={
                                            'filename': filename,
                                            'page': page_num,
                                            'table_index': table_idx + 1,
                                            'content_type': 'structured_table',
                                            'row_count': row_count,
                                            'upload_date': datetime.now().isoformat()
                                        }
                                    )
                                    table_chunks.append(chunk)

            logger.info(f"  ✓ Extracted {len(table_chunks)} valid tables (filtered garbage)")
        except Exception as e:
            logger.warning(f"  ⚠️ Table extraction failed: {e}")

        return table_chunks

    def ingest_document(self, filepath: str, filename: str, password: str = None, knowledge_base: str | None = None) -> dict:
        """Process and ingest document (PDF, Word, Excel) into vector store

        For PDFs: Automatically extracts tables as structured markdown + regular text
        """

        file_extension = Path(filename).suffix.lower()

        logger.info(f"⏳ Loading {file_extension} file: {filename}")

        all_chunks = []
        tables_extracted = 0

        # === PDF PROCESSING ===
        if file_extension == '.pdf':
            # STEP 1: Extract structured tables (enabled for academic documents)
            if settings.EXTRACT_TABLES:
                table_chunks = self._extract_tables_from_pdf(filepath, filename)
                all_chunks.extend(table_chunks)
                tables_extracted = len(table_chunks)
            else:
                logger.info("  ℹ️ Table extraction disabled")
                tables_extracted = 0

            # STEP 2: Extract regular text with PyMuPDF (for non-table content)
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                loader = PyMuPDFLoader(filepath)
                logger.info("  📄 Extracting text with PyMuPDF...")
            except ImportError:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(filepath)
                logger.info("  📄 Extracting text with PyPDF...")

            docs = loader.load()
            pages = len(docs)

            # Check if text extraction is empty (image-based PDF)
            total_text = sum(len(doc.page_content.strip()) for doc in docs)
            if total_text < 100:
                logger.warning(f"  ⚠️ PDF appears to be image-based (only {total_text} chars)")
                logger.info("  🔄 Attempting OCR extraction...")

                try:
                    from pdf2image import convert_from_path
                    import pytesseract

                    # Convert PDF to images and OCR (with password support)
                    pdf2image_kwargs = {}
                    if password:
                        # pdf2image uses 'userpw' parameter for password
                        pdf2image_kwargs['userpw'] = password
                        logger.info("    Using password for PDF conversion...")

                    images = convert_from_path(filepath, **pdf2image_kwargs)
                    ocr_docs = []

                    for i, image in enumerate(images):
                        logger.info(f"    OCR processing page {i+1}/{len(images)}...")
                        text = pytesseract.image_to_string(image, lang='eng')

                        if text.strip():  # Only add if text found
                            ocr_docs.append(Document(
                                page_content=text,
                                metadata={
                                    'filename': filename,
                                    'page': i + 1,
                                    'extraction_method': 'ocr'
                                }
                            ))

                    if ocr_docs:
                        docs = ocr_docs
                        pages = len(docs)
                        total_ocr_text = sum(len(d.page_content) for d in docs)
                        logger.info(f"  ✓ OCR extraction successful: {pages} pages, {total_ocr_text} chars")
                    else:
                        logger.error("  ❌ OCR found no text")
                        raise ValueError("Could not extract text from PDF (neither direct nor OCR)")

                except ImportError as e:
                    logger.error(f"  ❌ OCR libraries not installed: {e}")
                    logger.error("  💡 Install: pip install pytesseract pdf2image")
                    logger.error("  💡 Also install Tesseract: brew install tesseract poppler")
                    raise ValueError("PDF appears to be image-based but OCR not available")
                except Exception as e:
                    logger.error(f"  ❌ OCR failed: {e}")
                    raise ValueError(f"OCR extraction failed: {str(e)}")

            # Clean text and add metadata
            logger.info("  🧹 Cleaning text and extracting metadata...")

            # Extract document-level metadata from first few pages
            full_text = "\n".join([doc.page_content for doc in docs[:5]])
            logger.info(f"  🔍 Extracting metadata for: '{filename}'")
            doc_metadata = extract_document_metadata(full_text, filename)
            logger.info(f"  🔍 Extracted metadata: {doc_metadata}")

            if doc_metadata.get('title'):
                logger.info(f"  📋 Title: {doc_metadata.get('title')[:50]}...")
            if doc_metadata.get('authors'):
                logger.info(f"  � Authors: {doc_metadata.get('authors')[:50]}...")

            for i, doc in enumerate(docs):
                # Clean the text content
                doc.page_content = clean_academic_text(doc.page_content)

                # Add metadata
                doc.metadata['filename'] = filename
                doc.metadata['file_type'] = file_extension
                doc.metadata['content_type'] = 'text'
                doc.metadata['upload_date'] = datetime.now().isoformat()
                doc.metadata['page'] = i + 1
                doc.metadata['total_pages'] = pages

                # Add document-level metadata to each chunk
                doc.metadata.update(doc_metadata)

            # Filter out empty docs after cleaning
            docs = [d for d in docs if len(d.page_content.strip()) > 50]

            # Chunk text content
            text_chunks = self.text_splitter.split_documents(docs)

            # Add document title context if available and enabled
            if settings.PREPEND_CONTEXT and doc_metadata.get('title'):
                context_prefix = f"[{doc_metadata['title']}]\n\n"
                logger.info(f"  📌 Adding title prefix: {context_prefix.strip()}")

                prefixed_chunks = []
                for chunk in text_chunks:
                    new_content = context_prefix + chunk.page_content
                    new_chunk = Document(
                        page_content=new_content,
                        metadata=chunk.metadata.copy()
                    )
                    prefixed_chunks.append(new_chunk)

                logger.info(f"  ✅ Created {len(prefixed_chunks)} prefixed chunks")
                all_chunks.extend(prefixed_chunks)
            else:
                all_chunks.extend(text_chunks)
            logger.info(f"  ✓ Created {len(text_chunks)} text chunks (cleaned)")

        # === NON-PDF FILES ===
        elif file_extension in ['.docx', '.doc']:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(filepath)
            docs = loader.load()
            pages = len(docs)

            for i, doc in enumerate(docs):
                doc.metadata['filename'] = filename
                doc.metadata['file_type'] = file_extension
                doc.metadata['upload_date'] = datetime.now().isoformat()

            text_chunks = self.text_splitter.split_documents(docs)
            all_chunks.extend(text_chunks)

        elif file_extension in ['.xlsx', '.xls']:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(filepath, mode="elements")
            docs = loader.load()
            pages = len(docs)

            for i, doc in enumerate(docs):
                doc.metadata['filename'] = filename
                doc.metadata['file_type'] = file_extension
                doc.metadata['upload_date'] = datetime.now().isoformat()

            all_chunks.extend(docs)  # Excel elements are already chunked

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Add chunk-specific metadata
        for i, chunk in enumerate(all_chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        chunk_count = len(all_chunks)
        logger.info(f"📊 Total: {tables_extracted} tables + {chunk_count - tables_extracted} text chunks = {chunk_count} chunks")

        # Add to vectorstore（按知识库写入对应 collection）
        logger.info(f"⏳ Adding to vector database...")
        if all_chunks:
            store = self.get_store_for_knowledge_base(knowledge_base)
            store.add_documents(all_chunks)

        new_total = self.get_count(knowledge_base=knowledge_base)
        logger.info(f"✅ Total chunks in database: {new_total}")

        return {
            "tables_extracted": tables_extracted,
            "text_chunks": chunk_count - tables_extracted,
            "total_chunks_added": chunk_count,
            "total_in_database": new_total,
            "file_type": file_extension
        }
    
    def get_retriever(self, k: int = None, knowledge_base: str | None = None):
        """Get basic vector retriever"""
        k = k or settings.TOP_K
        store = self.get_store_for_knowledge_base(knowledge_base)
        return store.as_retriever(search_kwargs={"k": k})

    def clear_database(self, knowledge_base: str | None = None):
        """Clear all documents from vector store"""
        logger.info("🗑️  Clearing vector database...")
        # Get all IDs and delete them
        store = self.get_store_for_knowledge_base(knowledge_base)
        collection = store._collection
        results = collection.get()
        if results['ids']:
            collection.delete(ids=results['ids'])
        logger.info("✅ Database cleared")

# Global vector store manager
vector_store = VectorStoreManager()
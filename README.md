# RAG Automation - 智能文档问答系统

基于检索增强生成(RAG)的学术文档智能问答系统，专为大学课程材料、学术论文和技术文档设计。

## 🚀 功能特性

- **智能问答**：基于PDF、Word、Excel文档的自然语言问答
- **混合检索**：结合语义搜索和关键词搜索，提高检索准确性
- **重排序优化**：使用交叉编码器对检索结果进行精准排序
- **多格式支持**：支持PDF、DOCX、XLSX等多种文档格式
- **中文优化**：针对中文技术文档进行了专门优化
- **多知识库**：支持创建和管理多个知识库
- **环境变量配置**：灵活的数据目录配置
- **API接口**：提供完整的RESTful API接口
- **Web界面**：基于Streamlit的直观用户界面

## 🛠️ 技术栈

- **后端框架**：FastAPI + Python 3.12
- **前端界面**：Streamlit
- **向量数据库**：ChromaDB
- **AI模型**：
  - 嵌入模型：BAAI/bge-small-zh-v1.5
  - 大语言模型：Qwen2.5:7b (通过Ollama)
  - 重排序模型：BAAI/bge-reranker-base
- **核心组件**：LangChain、Sentence Transformers

## 📋 环境要求

- Python 3.12+
- Conda环境管理器
- Ollama (用于运行大语言模型)
- 至少8GB RAM
- Windows/Linux/macOS

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/xiaoxiaomanonglucifer/RAG_DOC.git
cd RAG_DOC-main
```

### 2. 创建Conda环境
```bash
# 创建新的conda环境
conda create -n RAG_DOC python=3.12 -y

# 激活环境
# Windows
conda activate RAG_DOC
# Linux/macOS
source activate RAG_DOC
```

### 3. 安装依赖
```bash
# 升级pip
pip install --upgrade pip

# 安装项目依赖
pip install -r config/requirements.txt
```

### 4. 安装Ollama和模型
```bash
# 安装Ollama (参考: https://ollama.ai/)
# 下载Qwen2.5模型
ollama pull qwen2.5:7b
```

### 5. 配置数据目录（可选）
系统默认数据存储在 `G:/RAG_DOC-main/RAG_DATA/`，如需修改：

**方式1：环境变量**
```bash
# Windows (PowerShell)
$env:RAG_DATA_ROOT=G:/RAG_Automation-main/RAG_DATA

# Linux/macOS
export RAG_DATA_ROOT=/home/user/rag_data
```

**方式2：修改.env文件**
```bash
# 在 config/.env 中添加
RAG_DATA_ROOT=G:/RAG_Automation-main/RAG_DATA
```

### 6. 启动服务

**启动后端API服务**
```bash
python main.py
```
(魔法)
访问http://127.0.0.1:8000/docs查看接口

**启动Web界面**（新终端）
```bash
streamlit run ui/gui_app.py
```

## 📁 项目结构

```
RAG_Automation-main/
├── work/                    # 核心业务逻辑
│   ├── __init__.py
│   ├── api.py             # FastAPI路由
│   ├── models.py          # 模型管理
│   ├── retrieval.py       # 检索逻辑
│   └── vector_store.py    # 向量存储
├── tools/                   # 工具和辅助功能
│   ├── __init__.py
│   ├── knowledge_base.py  # 知识库管理
│   ├── system_settings.py # 系统设置
│   └── openai_compat.py  # OpenAI兼容API
├── ui/                      # 用户界面
│   ├── __init__.py
│   ├── gui_app.py        # Streamlit主界面
│   └── chat_dialogue.py  # 对话功能
├── config/                  # 配置文件
│   ├── __init__.py
│   ├── settings.py        # 应用配置
│   ├── .env             # 环境变量
│   └── requirements.txt   # 依赖包
├── docs/                    # 文档
│   └── environment.md    # 环境配置说明
├── main.py                 # 应用入口
├── .gitignore             # Git忽略文件
└── README.md              # 项目说明

# 外部数据目录 (默认位置)
G:/RAG_DOC-main/RAG_DATA/
├── uploads/              # 临时上传文件
├── processed/            # 已处理文档
├── vector_db/            # ChromaDB向量数据库
├── chat_history/         # 对话历史记录
└── settings/            # 应用设置文件
```

## ⚙️ 配置说明

### 核心配置 (config/settings.py)

```python
# 数据目录配置
DATA_ROOT = "G:/RAG_DOC-main/RAG_DATA"  # 可通过环境变量RAG_DATA_ROOT覆盖

# 模型配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 嵌入模型
LLM_MODEL = "qwen2.5:7b"                    # 大语言模型
RERANKER_MODEL = "BAAI/bge-reranker-base"     # 重排序模型

# 检索配置
CHUNK_SIZE = 800               # 文档分块大小
CHUNK_OVERLAP = 200           # 分块重叠
TOP_K = 15                    # 返回结果数量
SEMANTIC_WEIGHT = 0.55         # 语义搜索权重
KEYWORD_WEIGHT = 0.45          # 关键词搜索权重

# API配置
API_HOST = "127.0.0.1"
API_PORT = 8000
```

### 环境变量 (config/.env)

```env
# HuggingFace模型缓存路径
HF_HOME=F:/HF_modals/huggingface
TRANSFORMERS_CACHE=F:/HF_modals/transformers
SENTENCE_TRANSFORMERS_HOME=F:/HF_modals/sentence-transformers


## 🎯 使用指南

### 1. 知识库管理
- 上传PDF、Word、Excel文档
- 查看已处理文档列表
- 删除不需要的文档
- 支持多个知识库管理

### 2. 智能对话
- 基于上传文档进行问答
- 支持流式输出
- 显示答案来源和引用
- 保存对话历史

### 3. 系统设置
- 配置API和Ollama服务地址
- 调整检索参数
- 选择LLM模型
- 设置界面主题


非GUI的完整执行流程
1. main.py (第30行) → uvicorn.run() 
   ↓
2. main.py (第22行) → from work.api import app
   ↓
3. work/api.py (第21-38行) → @asynccontextmanager lifespan()
   ↓
4. work/api.py (第27行) → model_manager.initialize()
   ↓
5. work/models.py (第115行) → ModelManager.initialize()
   ↓
6. work/api.py (第30行) → vector_store.initialize()
   ↓
7. work/api.py (第33行) → retriever.initialize()
   ↓
8. 启动完成，API服务就绪

main.py->api.py(models.py、vector_store.py、retriever.py) 
api.py里的代码学习fastapi框架

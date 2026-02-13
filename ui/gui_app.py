"""
RAG RCA Chatbot - Streamlit GUI Application
本地根因分析聊天机器人图形界面

功能特点：
- 三菜单布局：知识库管理、智能对话、系统设置
- 动态 Ollama 模型获取
- Streaming 流式输出
- 对话历史持久化
- 服务状态实时检测
"""

import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Generator

# 导入模块
from tools.knowledge_base import *
from ui.chat_dialogue import *
from tools.system_settings import *
from config.settings import settings

# ============================================================
# 配置常量
# ============================================================
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
HISTORY_FILE = Path(settings.CHAT_HISTORY_DIR) / "chat_history.json"
SETTINGS_FILE = Path(settings.SETTINGS_DIR) / "app_settings.json"

SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx", ".doc"]

# ============================================================
# Session State 初始化
# ============================================================
def init_session_state():
    """初始化所有 session state 变量"""

    # 加载保存的设置
    saved_settings = load_settings()

    defaults = {
        # 菜单导航
        "current_menu": "知识库管理",

        # 服务配置
        "api_url": saved_settings.get("api_url", DEFAULT_API_URL),
        "ollama_url": saved_settings.get("ollama_url", DEFAULT_OLLAMA_URL),

        # 对话相关
        "messages": [],                    # 聊天历史
        "selected_model": saved_settings.get("selected_model", ""),  # 当前模型
        "top_k": saved_settings.get("top_k", 5),
        "temperature": saved_settings.get("temperature", 0.7),
        "max_tokens": saved_settings.get("max_tokens", 512),
        "use_reranking": saved_settings.get("use_reranking", True),

        # 知识库相关
        "collections": [],                 # collections 列表（多知识库）
        "current_collection": "default",   # 当前选中的知识库名称
        "documents": [],                   # 文档列表

        # 状态
        "api_connected": False,
        "ollama_connected": False,
        "ollama_models": [],

        # 上传状态
        "upload_status": {},

        # 主题
        "theme": saved_settings.get("theme", "system"),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_settings() -> dict:
    """从文件加载设置"""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_settings():
    """保存设置到文件"""
    settings = {
        "api_url": st.session_state.api_url,
        "ollama_url": st.session_state.ollama_url,
        "selected_model": st.session_state.selected_model,
        "top_k": st.session_state.top_k,
        "temperature": st.session_state.temperature,
        "max_tokens": st.session_state.max_tokens,
        "use_reranking": st.session_state.use_reranking,
        "theme": st.session_state.theme,
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存设置失败: {e}")


def load_chat_history():
    """从文件加载对话历史"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_chat_history():
    """保存对话历史到文件"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存对话历史失败: {e}")


# ============================================================
# UI 渲染函数
# ============================================================
def render_status_bar():
    """渲染底部状态栏"""
    # 检查连接状态
    api_ok = st.session_state.api_connected
    ollama_ok = st.session_state.ollama_connected

    # 获取当前 collection 信息
    collection_info = "未选择"
    if st.session_state.collections:
        current_name = st.session_state.get("current_collection") or "default"
        col = next(
            (c for c in st.session_state.collections if c["name"] == current_name),
            st.session_state.collections[0],
        )
        collection_info = f"📁 {col['name']} | 📄 {col['document_count']} 文档 | 🧩 {col['chunk_count']} 块"

    # 获取当前模型
    model_info = st.session_state.selected_model or "未选择"

    # 渲染状态栏
    st.markdown("---")
    cols = st.columns([3, 2, 1, 1, 1])

    with cols[0]:
        st.markdown(f"**{collection_info}**")

    with cols[1]:
        st.markdown(f"**🤖 {model_info}**")

    with cols[2]:
        status = "🟢" if api_ok else "🔴"
        st.markdown(f"{status} FastAPI")

    with cols[3]:
        status = "🟢" if ollama_ok else "🔴"
        st.markdown(f"{status} Ollama")

    with cols[4]:
        st.markdown(f"v2.0")


# ============================================================
# 主应用
# ============================================================
def main():
    """主应用入口"""
    # 页面配置
    st.set_page_config(
        page_title="RAG RCA Chatbot",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定义 CSS
    st.markdown("""
    <style>
    /* 减少页面顶部空白，整体布局向上移动 */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* 用户消息样式 - 蓝色背景 */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #e3f2fd !important;
        border-left: 4px solid #2196f3;
    }
    
    /* AI 消息样式 - 绿色背景 */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #e8f5e9 !important;
        border-left: 4px solid #4caf50;
    }
    
    /* 代码样式 */
    code {
        background-color: #f5f5f5;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* 标题样式 */
    h1 {
        color: #1976d2;
    }
    
    /* 状态栏样式 */
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f8f9fa;
        padding: 10px 20px;
        border-top: 1px solid #dee2e6;
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

    # 初始化 session state
    init_session_state()

    # 加载对话历史
    if not st.session_state.messages:
        st.session_state.messages = load_chat_history()

    # 检查服务连接（仅在首次加载时）
    if "connection_checked" not in st.session_state:
        st.session_state.api_connected = check_api_health()
        st.session_state.ollama_connected = check_ollama_health()
        if st.session_state.ollama_connected:
            st.session_state.ollama_models = fetch_ollama_models()
        st.session_state.connection_checked = True

    # 页面标题
    st.title("🔧 RAG RCA Chatbot - 本地根因分析助手")
    st.markdown("基于本地文档的智能问答系统 | 支持混合检索 + 流式输出")
    st.divider()

    # 侧边栏菜单
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2>🔧 RAG RCA Chatbot</h2>
            <p style="color: #888; font-size: 14px;">本地根因分析助手</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # 简洁的主菜单
        st.markdown("""
        <style>
        .main-menu {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            border: 1px solid #e9ecef;
        }
        
        .menu-title {
            color: #495057;
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
        }
        
        /* 简洁的radio按钮样式 */
        .stRadio > div[role="radiogroup"] {
            background: transparent;
            border-radius: 8px;
            padding: 5px;
        }
        
        .stRadio > div[role="radiogroup"] > label {
            background: white;
            border-radius: 6px;
            padding: 10px 12px;
            margin: 3px 0;
            border: 1px solid #dee2e6;
            transition: all 0.2s ease;
            font-weight: 500;
            color: #495057;
        }
        
        .stRadio > div[role="radiogroup"] > label:hover {
            background: #f8f9fa;
            border-color: #667eea;
        }
        
        .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            color: #667eea;
        }
        
        .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div[aria-checked="true"] {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-menu">
            <div class="menu-title">🎯 主菜单</div>
        </div>
        """, unsafe_allow_html=True)
        
        menu = st.radio(
            "选择功能模块",
            options=["📚 知识库管理", "💬 智能对话", "⚙️ 系统设置"],
            index=["📚 知识库管理", "💬 智能对话", "⚙️ 系统设置"].index(st.session_state.current_menu) if st.session_state.current_menu in ["📚 知识库管理", "💬 智能对话", "⚙️ 系统设置"] else 0,
            label_visibility="collapsed"
        )
        
        # 更新session_state为简化的菜单名
        menu_mapping = {
            "📚 知识库管理": "知识库管理",
            "💬 智能对话": "智能对话", 
            "⚙️ 系统设置": "系统设置"
        }
        st.session_state.current_menu = menu_mapping[menu]

        st.divider()

        # 快捷信息
        st.subheader("📊 系统状态")

        api_status = "🟢 已连接" if st.session_state.api_connected else "🔴 未连接"
        ollama_status = "🟢 已连接" if st.session_state.ollama_connected else "🔴 未连接"

        st.markdown(f"**FastAPI**: {api_status}")
        st.markdown(f"**Ollama**: {ollama_status}")

        if st.session_state.collections:
            current_name = st.session_state.get("current_collection") or "default"
            col_data = next(
                (c for c in st.session_state.collections if c["name"] == current_name),
                st.session_state.collections[0],
            )
            st.markdown(f"**当前知识库**: `{col_data.get('name', 'default')}`")
            st.markdown(f"**文档数**: {col_data.get('document_count', 0)}")
            st.markdown(f"**块数**: {col_data.get('chunk_count', 0)}")

        st.divider()

        # 底部信息
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 12px;">
            RAG RCA Chatbot v2.0<br>
            Powered by FastAPI + Ollama + ChromaDB
        </div>
        """, unsafe_allow_html=True)

    # 根据菜单渲染对应页面
    if st.session_state.current_menu == "知识库管理":
        render_knowledge_base_page()
        # 渲染底部状态栏
        render_status_bar()
    elif st.session_state.current_menu == "智能对话":
        render_chat_page()
    elif st.session_state.current_menu == "系统设置":
        render_settings_page()
        # 渲染底部状态栏
        render_status_bar()


if __name__ == "__main__":
    main()
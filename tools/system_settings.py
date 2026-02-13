"""
System Settings Module
系统设置模块

包含所有系统设置相关的功能：
- 设置文件管理
- 服务健康检查
- 系统配置界面
"""

import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from config.settings import settings

# ============================================================
# 常量配置
# ============================================================
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
SETTINGS_FILE =Path(settings.SETTINGS_DIR) / "rag_automation_settings.json"

# ============================================================
# 设置文件管理
# ============================================================
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

# ============================================================
# 服务健康检查
# ============================================================
def check_api_health() -> bool:
    """检查 FastAPI 后端健康状态"""
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_ollama_health() -> bool:
    """检查 Ollama 服务健康状态"""
    try:
        response = requests.get(f"{st.session_state.ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def fetch_ollama_models() -> List[str]:
    """从 Ollama 获取本地模型列表"""
    try:
        response = requests.get(f"{st.session_state.ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models)
    except Exception:
        pass
    return []


def fetch_system_status() -> Dict:
    """获取系统状态"""
    try:
        response = requests.get(f"{st.session_state.api_url}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}

# ============================================================
# UI 渲染函数
# ============================================================
def render_settings_page():
    """渲染系统设置页面 - 子菜单导航设计"""

    # 添加页面级别的CSS样式
    st.markdown("""
    <style>
    /* 主容器样式 */
    .settings-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }

    .settings-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(-20px, -20px) rotate(180deg); }
    }

    /* 功能卡片 */
    .settings-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .settings-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }

    /* 状态卡片 */
    .status-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .status-card.warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    .status-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .status-icon {
        font-size: 2em;
        margin-bottom: 10px;
    }

    .status-text {
        font-size: 1.1em;
        font-weight: bold;
        margin: 0;
    }

    /* 隐藏标题边框 */
    .settings-card h3,
    .stMarkdown h3,
    .element-container h3,
    h1, h2, h3, h4, h5, h6 {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 0 15px 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* 动画效果 */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: slideIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

    # 子菜单导航
    # st.markdown('<div class="settings-header">', unsafe_allow_html=True)
    st.markdown('<h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">⚙️ 系统设置中心</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 子菜单选择
    sub_menu = st.selectbox(
        "🎯 选择功能模块",
        options=["🌐 网络服务", "🤖 AI模型", "💾 数据存储", "📊 系统监控"],
        help="选择要配置的功能模块"
    )

    # 根据选择渲染对应功能
    if sub_menu == "🌐 网络服务":
        render_network_services()
    elif sub_menu == "🤖 AI模型":
        render_ai_model_config()
    elif sub_menu == "💾 数据存储":
        render_data_storage()
    elif sub_menu == "📊 系统监控":
        render_system_monitoring()


def render_network_services():
    """渲染网络服务配置"""
    st.markdown("### 🌐 网络服务")

    # 服务地址配置
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔗 FastAPI 后端")
        new_api_url = st.text_input(
            "服务地址",
            value=st.session_state.api_url,
            help="FastAPI 服务的完整 URL，例如: http://localhost:8000"
        )

        # 连接状态
        api_status = "🟢 已连接" if st.session_state.api_connected else "🔴 未连接"
        status_class = "" if st.session_state.api_connected else "warning"
        st.markdown(f"""
        <div class="status-card {status_class}">
            <div class="status-icon">{api_status.split()[0]}</div>
            <div class="status-text">{api_status.split()[1]} {api_status.split()[2] if len(api_status.split()) > 2 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🤖 Ollama 服务")
        new_ollama_url = st.text_input(
            "服务地址",
            value=st.session_state.ollama_url,
            help="Ollama 服务的完整 URL，例如: http://localhost:11434"
        )

        # 连接状态
        ollama_status = "🟢 已连接" if st.session_state.ollama_connected else "🔴 未连接"
        status_class = "" if st.session_state.ollama_connected else "warning"
        st.markdown(f"""
        <div class="status-card {status_class}">
            <div class="status-icon">{ollama_status.split()[0]}</div>
            <div class="status-text">{ollama_status.split()[1]} {ollama_status.split()[2] if len(ollama_status.split()) > 2 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    # 测试连接按钮
    st.markdown("#### 🔄 连接测试")
    col_test1, col_test2 = st.columns(2)

    with col_test1:
        if st.button("🔄 测试 FastAPI 连接", use_container_width=True):
            st.session_state.api_url = new_api_url
            with st.spinner("正在测试连接..."):
                if check_api_health():
                    st.success("✅ FastAPI 连接成功！")
                    st.session_state.api_connected = True
                else:
                    st.error(f"❌ 无法连接到 FastAPI: {new_api_url}")
                    st.session_state.api_connected = False

    with col_test2:
        if st.button("🔄 测试 Ollama 连接", use_container_width=True):
            st.session_state.ollama_url = new_ollama_url
            with st.spinner("正在测试连接..."):
                if check_ollama_health():
                    st.success("✅ Ollama 连接成功！")
                    st.session_state.ollama_connected = True
                    st.session_state.ollama_models = fetch_ollama_models()
                else:
                    st.error(f"❌ 无法连接到 Ollama: {new_ollama_url}")
                    st.session_state.ollama_connected = False

    # 保存服务配置
    if st.button("💾 保存服务配置", type="primary", use_container_width=True):
        st.session_state.api_url = new_api_url
        st.session_state.ollama_url = new_ollama_url
        save_settings()
        st.success("✅ 服务配置已保存！")

    st.markdown('</div>', unsafe_allow_html=True)


def render_ai_model_config():
    """渲染AI模型配置"""
    # st.markdown('<div class="settings-card animate-in">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI模型配置")

    # 模型选择
    st.markdown("#### 🎯 模型选择")
    models = st.session_state.ollama_models
    if not models:
        models = ["qwen2.5:7b", "llama3.2:8b", "deepseek-r1:7b"]

    selected_model = st.selectbox(
        "选择 Ollama 模型",
        options=models,
        index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0,
        help="选择用于对话的AI模型"
    )
    st.session_state.selected_model = selected_model

    # 刷新模型列表
    if st.button("🔄 刷新模型列表", use_container_width=True):
        with st.spinner("正在刷新模型列表..."):
            st.session_state.ollama_models = fetch_ollama_models()
        st.success("✅ 模型列表已刷新！")
        st.rerun()

    # 参数配置
    st.markdown("#### ⚙️ 参数调优")
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.top_k = st.slider(
            "检索数量 (top_k)",
            min_value=3,
            max_value=20,
            value=st.session_state.top_k,
            help="从向量数据库检索的相关文档块数量"
        )

        st.session_state.temperature = st.slider(
            "温度 (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="0=确定性回答，1=创造性回答"
        )

    with col2:
        st.session_state.max_tokens = st.number_input(
            "最大 Token 数",
            min_value=256,
            max_value=4096,
            value=st.session_state.max_tokens,
            step=256,
            help="生成回答的最大长度"
        )

        st.session_state.use_reranking = st.checkbox(
            "启用重排序",
            value=st.session_state.use_reranking,
            help="使用交叉编码器对检索结果重新排序，提高准确性"
        )

    # 保存模型配置
    if st.button("💾 保存模型配置", type="primary", use_container_width=True):
        save_settings()
        st.success("✅ 模型配置已保存！")

    st.markdown('</div>', unsafe_allow_html=True)


def render_data_storage():
    """渲染数据存储管理"""
    st.markdown("### 💾 数据存储管理")

    # 数据概览
    st.markdown("#### 📊 数据概览")
    col1, col2 = st.columns(2)

    with col1:
        # 对话历史信息
        from ui.chat_dialogue import HISTORY_FILE
        history_size = len(st.session_state.messages)
        history_file_size = f"{HISTORY_FILE.stat().st_size / 1024:.1f} KB" if HISTORY_FILE.exists() else "0 KB"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">💬</div>
            <div class="status-text">对话历史</div>
            <div style="font-size: 0.9em; opacity: 0.9;">{history_size} 条消息 | {history_file_size}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 设置文件信息
        settings_file_size = f"{SETTINGS_FILE.stat().st_size / 1024:.1f} KB" if SETTINGS_FILE.exists() else "0 KB"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-icon">⚙️</div>
            <div class="status-text">系统设置</div>
            <div style="font-size: 0.9em; opacity: 0.9;">配置文件 | {settings_file_size}</div>
        </div>
        """, unsafe_allow_html=True)

    # 数据管理操作
    st.markdown("#### 🛠️ 数据管理")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**对话历史管理**")
        from ui.chat_dialogue import HISTORY_FILE
        st.markdown(f"文件路径: `{HISTORY_FILE}`")

        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ 清除历史", use_container_width=True, help="清除所有对话历史"):
                st.session_state.messages = []
                if HISTORY_FILE.exists():
                    HISTORY_FILE.unlink()
                st.success("✅ 对话历史已清除")
                st.rerun()

        with col_export:
            if st.button("📤 导出历史", use_container_width=True, help="导出对话历史为JSON文件"):
                if st.session_state.messages:
                    export_data = {
                        "export_time": datetime.now().isoformat(),
                        "messages": st.session_state.messages
                    }
                    st.download_button(
                        label="💾 下载文件",
                        data=json.dumps(export_data, ensure_ascii=False, indent=2),
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("📭 暂无对话历史可导出")

    with col2:
        st.markdown("**设置文件管理**")
        st.markdown(f"文件路径: `{SETTINGS_FILE}`")

        col_reset, col_backup = st.columns(2)
        with col_reset:
            if st.button("🔄 重置设置", use_container_width=True, help="重置所有设置为默认值"):
                if SETTINGS_FILE.exists():
                    SETTINGS_FILE.unlink()
                st.success("✅ 设置已重置，刷新页面后生效")

        with col_backup:
            if st.button("💾 备份设置", use_container_width=True, help="备份当前设置"):
                if SETTINGS_FILE.exists():
                    with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                        settings_data = f.read()
                    st.download_button(
                        label="💾 下载备份",
                        data=settings_data,
                        file_name=f"settings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("📭 暂无设置文件可备份")

    # 数据导入
    st.markdown("#### 📥 数据导入")
    uploaded_file = st.file_uploader(
        "上传设置备份文件",
        type=['json'],
        help="上传之前备份的设置文件进行恢复"
    )

    if uploaded_file:
        try:
            import_data = json.loads(uploaded_file.read().decode('utf-8'))
            if st.button("🔄 恢复设置", type="primary", use_container_width=True):
                with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(import_data, f, ensure_ascii=False, indent=2)
                st.success("✅ 设置已恢复，刷新页面后生效")
        except Exception as e:
            st.error(f"❌ 文件格式错误: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_system_monitoring():
    """渲染系统监控"""
    st.markdown("### 📊 系统监控")

    # 服务状态仪表板
    st.markdown("#### 🖥️ 服务状态")
    col1, col2 = st.columns(2)

    with col1:
        # FastAPI 状态
        api_status = st.session_state.api_connected
        status_color = "#43e97b" if api_status else "#f5576c"
        status_text = "🟢 正常运行" if api_status else "🔴 服务异常"

        st.markdown(f"""
        <div style="background: {status_color}; border-radius: 15px; padding: 20px; text-align: center; color: white;">
            <div style="font-size: 2em; margin-bottom: 10px;">🌐</div>
            <div style="font-size: 1.2em; font-weight: bold;">FastAPI 服务</div>
            <div style="font-size: 1em; opacity: 0.9;">{status_text}</div>
            <div style="font-size: 0.9em; opacity: 0.8; margin-top: 5px;">{st.session_state.api_url}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Ollama 状态
        ollama_status = st.session_state.ollama_connected
        status_color = "#43e97b" if ollama_status else "#f5576c"
        status_text = "🟢 正常运行" if ollama_status else "🔴 服务异常"
        model_count = len(st.session_state.ollama_models)

        st.markdown(f"""
        <div style="background: {status_color}; border-radius: 15px; padding: 20px; text-align: center; color: white;">
            <div style="font-size: 2em; margin-bottom: 10px;">🤖</div>
            <div style="font-size: 1.2em; font-weight: bold;">Ollama 服务</div>
            <div style="font-size: 1em; opacity: 0.9;">{status_text}</div>
            <div style="font-size: 0.9em; opacity: 0.8; margin-top: 5px;">{model_count} 个可用模型</div>
        </div>
        """, unsafe_allow_html=True)

    # 系统信息
    st.markdown("#### ℹ️ 系统信息")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**应用信息**")
        st.markdown(f"- **版本**: v2.0")
        import sys
        st.markdown(f"- **Python**: {sys.version.split()[0]}")
        st.markdown(f"- **当前知识库**: {st.session_state.current_collection}")

        if st.session_state.collections:
            current = next((c for c in st.session_state.collections if c["name"] == st.session_state.current_collection), None)
            if current:
                st.markdown(f"- **文档数量**: {current.get('document_count', 0)}")
                st.markdown(f"- **数据块数量**: {current.get('chunk_count', 0)}")

    with col2:
        st.markdown("**配置信息**")
        st.markdown(f"- **API地址**: {st.session_state.api_url}")
        st.markdown(f"- **Ollama地址**: {st.session_state.ollama_url}")
        st.markdown(f"- **当前模型**: {st.session_state.selected_model or '未选择'}")
        st.markdown(f"- **重排序**: {'启用' if st.session_state.use_reranking else '禁用'}")

    # 性能指标
    st.markdown("#### 📈 性能指标")

    # 模拟性能数据（实际应用中可以从真实API获取）
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; padding: 20px; text-align: center; color: white;">
            <div style="font-size: 2em; margin-bottom: 10px;">⚡</div>
            <div style="font-size: 1.5em; font-weight: bold;">正常</div>
            <div style="font-size: 0.9em; opacity: 0.9;">响应速度</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; padding: 20px; text-align: center; color: white;">
            <div style="font-size: 2em; margin-bottom: 10px;">💾</div>
            <div style="font-size: 1.5em; font-weight: bold;">正常</div>
            <div style="font-size: 0.9em; opacity: 0.9;">内存使用</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 15px; padding: 20px; text-align: center; color: white;">
            <div style="font-size: 2em; margin-bottom: 10px;">🔄</div>
            <div style="font-size: 1.5em; font-weight: bold;">正常</div>
            <div style="font-size: 0.9em; opacity: 0.9;">系统负载</div>
        </div>
        """, unsafe_allow_html=True)

    # 刷新监控数据
    if st.button("🔄 刷新监控数据", use_container_width=True):
        with st.spinner("正在刷新监控数据..."):
            # 重新检查服务状态
            st.session_state.api_connected = check_api_health()
            st.session_state.ollama_connected = check_ollama_health()
            if st.session_state.ollama_connected:
                st.session_state.ollama_models = fetch_ollama_models()
            from .knowledge_base import fetch_collections
            st.session_state.collections = fetch_collections()
        st.success("✅ 监控数据已刷新！")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

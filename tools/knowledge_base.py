"""
Knowledge Base Management Module
知识库管理模块

包含所有知识库相关的功能：
- 知识库列表获取
- 文档上传和删除
- 知识库管理界面
"""

import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# ============================================================
# 常量配置
# ============================================================
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx", ".doc"]

# ============================================================
# API 客户端函数
# ============================================================
def fetch_collections() -> List[Dict]:
    """
    获取所有知识库及其文档列表。

    后端现在支持多知识库：
    - /knowledge_bases 返回知识库名称列表
    - /documents?knowledge_base=xxx 返回对应知识库的文档明细
    """
    collections: List[Dict] = []
    try:
        kb_resp = requests.get(f"{st.session_state.api_url}/knowledge_bases", timeout=10)
        if kb_resp.status_code != 200:
            st.error(f"获取知识库列表失败: HTTP {kb_resp.status_code}")
            return []

        kb_data = kb_resp.json()
        kb_items = kb_data.get("knowledge_bases", [])
        if not kb_items:
            return []

        for item in kb_items:
            name = item.get("name", "default")
            try:
                docs_resp = requests.get(
                    f"{st.session_state.api_url}/documents",
                    params={"knowledge_base": name},
                    timeout=10,
                )
                if docs_resp.status_code == 200:
                    docs_data = docs_resp.json()
                else:
                    docs_data = {}
            except Exception:
                docs_data = {}

            collections.append({
                "name": name,
                "document_count": docs_data.get("total_documents", 0),
                "chunk_count": docs_data.get("total_chunks", 0),
                "documents": docs_data.get("documents", []),
            })

        # 默认知识库优先显示
        collections.sort(key=lambda c: (0 if c["name"] == "default" else 1, c["name"]))
        return collections

    except Exception as e:
        st.error(f"获取知识库列表失败: {e}")
        return []


def upload_file_to_api(file, password: Optional[str] = None, knowledge_base: Optional[str] = None) -> Dict:
    """上传文件到后端"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        kb = knowledge_base or st.session_state.current_collection or "default"
        data = {"knowledge_base": kb}
        if password:
            data["password"] = password

        response = requests.post(
            f"{st.session_state.api_url}/ingest",
            files=files,
            data=data,
            timeout=120
        )

        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "detail": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def delete_document(filename: str, knowledge_base: Optional[str] = None) -> bool:
    """删除指定文档"""
    try:
        params = {}
        kb = knowledge_base or st.session_state.current_collection or "default"
        params["knowledge_base"] = kb

        response = requests.delete(
            f"{st.session_state.api_url}/documents/{filename}",
            params=params,
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"删除文档失败: {e}")
        return False

# ============================================================
# UI 渲染函数
# ============================================================
def render_knowledge_base_page():
    """渲染知识库管理页面 - 子菜单导航设计"""
    # st.header("📚 知识库管理")

    # 添加页面级别的CSS样式
    st.markdown("""
    <style>
    /* 主容器样式 */
    .main-header {
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

    .main-header::before {
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
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }

    /* 统计卡片 */
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .stat-card:nth-child(2) {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .stat-card:nth-child(3) {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }

    .stat-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .stat-number {
        font-size: 3em;
        font-weight: bold;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .stat-label {
        font-size: 1em;
        opacity: 0.9;
        margin: 10px 0 0 0;
    }

    /* 文档项 */
    .doc-item {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        color: #333;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .doc-item::before {
        content: '📄';
        position: absolute;
        left: -20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 2em;
        opacity: 0.3;
    }

    .doc-item:hover {
        transform: translateX(10px);
        box-shadow: 0 5px 20px rgba(252, 182, 159, 0.4);
    }

    /* 上传区域 */
    .upload-area {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: 3px dashed #4CAF50;
        border-radius: 20px;
        padding: 50px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .upload-area:hover {
        border-color: #45a049;
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
    }

    /* 知识库卡片 */
    .kb-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .kb-card:hover {
        border-left-width: 8px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }

    /* 隐藏标题边框 */
    .feature-card h3,
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

    # 确保有最新的 collections
    if not st.session_state.collections:
        st.session_state.collections = fetch_collections()

    collections = st.session_state.collections
    kb_names = [c["name"] for c in collections] if collections else ["default"]

    if st.session_state.current_collection not in kb_names:
        st.session_state.current_collection = "default" if "default" in kb_names else kb_names[0]

    # 子菜单导航
    st.markdown('<h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">📚 知识库管理中心</h1>', unsafe_allow_html=True)
    # st.markdown('</div>', unsafe_allow_html=True)

    # 子菜单选择
    sub_menu = st.selectbox(
        "🎯 选择功能模块",
        options=["📚 知识库概览", "📄 文档中心", "🗂️ 知识库管理"],
        help="选择要使用的功能模块"
    )

    # 根据选择渲染对应功能
    if sub_menu == "📚 知识库概览":
        render_kb_overview(collections, kb_names)
    elif sub_menu == "📄 文档中心":
        render_document_center(collections)
    elif sub_menu == "🗂️ 知识库管理":
        render_kb_management(collections, kb_names)


def render_kb_overview(collections, kb_names):
    """渲染知识库概览页面"""
    st.markdown("### 📚 知识库概览")

    # 知识库选择
    col_selector, col_refresh = st.columns([3, 1])

    with col_selector:
        selected_kb = st.selectbox(
            "🎯 选择知识库",
            options=kb_names,
            index=kb_names.index(st.session_state.current_collection) if kb_names else 0,
            help="选择要查看的知识库"
        )
        st.session_state.current_collection = selected_kb

    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄", help="刷新数据", use_container_width=True):
            with st.spinner("🔄 正在刷新..."):
                st.session_state.collections = fetch_collections()
            st.success("✅ 数据已更新")
            st.rerun()

    if collections:
        current = next((c for c in collections if c["name"] == st.session_state.current_collection), collections[0])

        # 统计信息
        st.markdown("#### 📊 统计信息")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{current.get("document_count", 0)}</div>
                <div class="stat-label">📄 文档总数</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{current.get("chunk_count", 0)}</div>
                <div class="stat-label">🧩 数据块数</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            status_color = "#43e97b" if current.get("document_count", 0) > 0 else "#f5576c"
            status_text = "✅ 活跃" if current.get("document_count", 0) > 0 else "⚠️ 空库"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{status_text}</div>
                <div class="stat-label">📊 库状态</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_document_center(collections):
    """渲染文档中心页面"""
    if not collections:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-radius: 20px; margin: 20px 0;">
            <h3 style="color: #333;">📭 暂无知识库</h3>
            <p style="color: #666;">请先创建知识库再管理文档</p>
        </div>
        """, unsafe_allow_html=True)
        return

    current = next((c for c in collections if c["name"] == st.session_state.current_collection), collections[0])

    st.markdown("### 📄 文档中心")

    # 标签页切换
    tab1, tab2 = st.tabs(["📋 文档管理", "📤 文档上传"])

    with tab1:
        documents = current.get("documents", [])

        if documents:
            for doc in documents:
                with st.container():
                    st.markdown(f"""
                    <div class="doc-item">
                        <div>
                            <strong>{doc.get('filename', 'Unknown')}</strong><br>
                            <small>🧩 {doc.get('chunks', 0)} 块 | 📅 {doc.get('upload_date', 'Unknown')}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    delete_key = f"delete_{doc.get('filename', '')}"
                    if st.button("🗑️", key=delete_key, help=f"删除 {doc.get('filename', '')}"):
                        # 确认删除对话框
                        st.warning(f"⚠️ 确认要删除文档 '{doc.get('filename', '')}' 吗？")
                        col_confirm, col_cancel = st.columns(2)
                        
                        with col_confirm:
                            if st.button(f"✅ 确认删除", key=f"confirm_del_{doc.get('filename', '')}", type="primary"):
                                if delete_document(doc.get('filename', ''), knowledge_base=st.session_state.current_collection):
                                    st.success(f"✅ 已删除: {doc.get('filename', '')}")
                                    time.sleep(0.5)
                                    st.session_state.collections = fetch_collections()
                                    st.rerun()
                        
                        with col_cancel:
                            if st.button(f"❌ 取消", key=f"cancel_del_{doc.get('filename', '')}"):
                                st.rerun()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 3em; margin-bottom: 15px;">📭</div>
                <p>当前知识库暂无文档</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "🎯 拖拽或点击选择文件",
            type=["pdf", "txt", "md", "docx", "doc"],
            accept_multiple_files=True,
            help="支持 PDF, TXT, MD, DOCX, DOC 格式",
            label_visibility="collapsed"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # PDF密码
        pdf_password = None
        if uploaded_files:
            has_pdf = any(f.name.lower().endswith('.pdf') for f in uploaded_files)
            if has_pdf:
                with st.expander("🔐 PDF 密码保护", expanded=True):
                    pdf_password = st.text_input(
                        "密码",
                        type="password",
                        help="如果PDF有密码保护，请输入",
                        placeholder="输入密码..."
                    )

        # 上传按钮
        if uploaded_files:
            if st.button("🚀 开始上传", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                success_count = 0

                for i, file in enumerate(uploaded_files):
                    progress = (i) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.markdown(f"📤 **{file.name}** ({i+1}/{len(uploaded_files)})")

                    result = upload_file_to_api(file, pdf_password, knowledge_base=st.session_state.current_collection)

                    if result.get("status") == "success":
                        success_count += 1
                    else:
                        st.error(f"❌ {file.name}: {result.get('detail', '未知错误')}")

                progress_bar.progress(1.0)
                status_text.empty()

                if success_count > 0:
                    st.success(f"🎉 成功上传 {success_count} 个文件！")

                time.sleep(0.5)
                st.session_state.collections = fetch_collections()
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_kb_management(collections, kb_names):
    """渲染知识库管理页面"""
    st.markdown("### 🗂️ 知识库管理")

    # 标签页切换
    tab1, tab2 = st.tabs(["➕ 创建知识库", "📋 知识库列表"])

    with tab1:
        with st.form("create_kb_form", clear_on_submit=True):
            st.markdown("#### ✨ 创建新知识库")

            new_kb_name = st.text_input(
                "知识库名称",
                placeholder="输入新知识库名称...",
                help="为您的知识库取一个有意义的名称"
            )

            submitted = st.form_submit_button("🎯 创建知识库", type="primary", use_container_width=True)

            if submitted:
                kb_name = (new_kb_name or "").strip()
                if kb_name:
                    # 确认创建对话框
                    st.info(f"ℹ️ 确认要创建新知识库 '{kb_name}' 吗？")
                    col_confirm, col_cancel = st.columns(2)
                    
                    with col_confirm:
                        if st.button(f"✅ 确认创建", key=f"confirm_create_{kb_name}", type="primary"):
                            try:
                                resp = requests.post(
                                    f"{st.session_state.api_url}/knowledge_bases",
                                    json={"name": kb_name},
                                    timeout=10,
                                )
                                if resp.status_code == 200:
                                    st.success(f"🎉 知识库 '{kb_name}' 创建成功！")
                                    st.session_state.current_collection = kb_name
                                    time.sleep(0.5)
                                    st.session_state.collections = fetch_collections()
                                    st.rerun()
                                else:
                                    st.error(f"❌ 创建失败: HTTP {resp.status_code}")
                            except Exception as e:
                                st.error(f"❌ 创建失败: {e}")
                    
                    with col_cancel:
                        if st.button(f"❌ 取消", key=f"cancel_create_{kb_name}"):
                            st.rerun()
                else:
                    st.error("⚠️ 请输入知识库名称")

    with tab2:
        if collections:
            st.markdown("#### 📋 现有知识库")

            # 网格显示知识库
            cols = st.columns(min(3, len(collections)))
            for i, kb in enumerate(collections):
                with cols[i % len(cols)]:
                    is_current = kb['name'] == st.session_state.current_collection
                    bg_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" if is_current else "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"

                    st.markdown(f"""
                    <div class="kb-card">
                        <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 10px;">📁 {kb['name']}</div>
                        <div style="color: #666; margin-bottom: 10px;">📄 {kb['document_count']} 文档</div>
                        <div style="color: #888; font-size: 0.9em;">{"✅ 当前使用" if is_current else "⚪ 可选择"}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if kb['name'] != 'default':
                        if st.button("🗑️ 删除", key=f"del_kb_{kb['name']}", help=f"删除 {kb['name']}", use_container_width=True):
                            # 确认删除对话框
                            st.warning(f"⚠️ 确认要删除知识库 '{kb['name']}' 吗？")
                            col_confirm, col_cancel = st.columns(2)
                            
                            with col_confirm:
                                if st.button(f"✅ 确认删除", key=f"confirm_del_kb_{kb['name']}", type="primary"):
                                    try:
                                        resp = requests.delete(f"{st.session_state.api_url}/knowledge_bases/{kb['name']}")
                                        if resp.status_code == 200:
                                            st.success(f"✅ 知识库 '{kb['name']}' 已删除")
                                            if st.session_state.current_collection == kb['name']:
                                                st.session_state.current_collection = "default"
                                            st.session_state.collections = fetch_collections()
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.error("删除失败")
                                    except Exception as e:
                                        st.error(f"删除失败: {e}")
                            
                            with col_cancel:
                                if st.button(f"❌ 取消", key=f"cancel_del_kb_{kb['name']}"):
                                    st.rerun()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 3em; margin-bottom: 15px;">📭</div>
                <p>暂无知识库，请先创建</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

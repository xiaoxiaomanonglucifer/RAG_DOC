import streamlit as st
import json
import time
import requests
from datetime import datetime
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

# 常量定义
HISTORY_FILE = Path(settings.CHAT_HISTORY_DIR) / "chat_history.json"

def save_chat_history():
    """保存对话历史到文件"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存历史失败: {e}")

def load_chat_history():
    """从文件加载对话历史"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def query_rag(question: str, knowledge_base: str = "default", use_reranking: bool = True, include_context: bool = False):
    """调用RAG API进行查询"""
    try:
        # 检查API连接状态
        if not st.session_state.get("api_connected", False):
            return {
                "answer": "❌ API服务未连接，请先检查服务状态。",
                "sources": [],
                "error": "API未连接"
            }
        
        # 构建请求参数 - 发送JSON数据
        data = {
            "question": question,
            "knowledge_base": knowledge_base,
            "use_reranking": use_reranking,  # 保持布尔类型
            "include_context": include_context  # 保持布尔类型
        }
        
        # 发送请求 - 使用JSON数据
        response = requests.post(
            f"{st.session_state.api_url}/query",
            json=data,  # 使用json而不是data
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"❌ 查询失败，HTTP状态码: {response.status_code}",
                "sources": [],
                "error": f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {
            "answer": "❌ 请求超时，请稍后重试。",
            "sources": [],
            "error": "请求超时"
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "❌ 无法连接到API服务，请检查服务是否正常运行。",
            "sources": [],
            "error": "连接错误"
        }
    except Exception as e:
        return {
            "answer": f"❌ 查询过程中发生错误: {str(e)}",
            "sources": [],
            "error": str(e)
        }

def render_chat_page():
    """渲染聊天页面"""
    # 子菜单导航
    st.markdown('<h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">💬 智能对话系统</h1>', unsafe_allow_html=True)
    
    # 子菜单选择
    sub_menu = st.selectbox(
        "🎯 选择功能模块",
        options=["💭 对话界面", "📚 对话历史"],
        help="选择要使用的功能模块"
    )
    
    # 根据选择渲染对应功能
    if sub_menu == "💭 对话界面":
        render_chat_interface()
    elif sub_menu == "📚 对话历史":
        render_chat_history()

def render_chat_interface():
    """渲染聊天界面 - 现代微信风格设计"""
    st.markdown("### 💭 智能对话")
    
    # 添加自定义CSS样式
    st.markdown("""
    <style>
    /* 聊天容器样式 */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px 0;
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
    }
    
    /* 用户消息样式 */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
        animation: slideInRight 0.3s ease-out;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        box-shadow: 0 3px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        word-wrap: break-word;
    }
    
    .user-avatar {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: 10px;
        font-size: 18px;
        box-shadow: 0 3px 15px rgba(102, 126, 234, 0.3);
        flex-shrink: 0;
    }
    
    /* AI消息样式 */
    .ai-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 20px;
        animation: slideInLeft 0.3s ease-out;
    }
    
    .ai-bubble {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #333;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        position: relative;
        word-wrap: break-word;
    }
    
    .ai-avatar {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 18px;
        box-shadow: 0 3px 15px rgba(40, 167, 69, 0.3);
        flex-shrink: 0;
    }
    
    /* 消息时间戳 */
    .message-time {
        font-size: 11px;
        opacity: 0.7;
        margin-top: 8px;
        color: #6c757d;
        font-weight: 500;
    }
    
    /* 来源信息样式 */
    .sources-container {
        margin-top: 12px;
        padding: 12px;
        background: rgba(40, 167, 69, 0.1);
        border-radius: 8px;
        border-left: 3px solid #28a745;
    }
    
    .source-title {
        font-weight: bold;
        color: #28a745;
        margin-bottom: 8px;
        font-size: 12px;
    }
    
    .source-item {
        font-size: 11px;
        margin-bottom: 6px;
        padding: 6px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 4px;
        border-left: 2px solid #20c997;
    }
    
    .source-filename {
        font-weight: 600;
        color: #495057;
    }
    
    .source-details {
        color: #6c757d;
        font-size: 10px;
    }
    
    /* 动画效果 */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .user-bubble, .ai-bubble {
            max-width: 85%;
        }
        .user-avatar, .ai-avatar {
            width: 35px;
            height: 35px;
            font-size: 16px;
        }
    }
    
    /* 隐藏Streamlit默认样式 */
    .stChatMessage {
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 聊天容器
    with st.container():
        # 显示聊天消息 - 垂直时间线布局
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                # 用户消息
                st.markdown(f"""
                <div class="user-message">
                    <div class="user-bubble">
                        <div>{message["content"]}</div>
                        <div class="message-time">👤 用户</div>
                    </div>
                    <div class="user-avatar">👤</div>
                </div>
                """, unsafe_allow_html=True)
                
            elif message["role"] == "assistant":
                # AI回复
                st.markdown(f"""
                <div class="ai-message">
                    <div class="ai-avatar">🤖</div>
                    <div class="ai-bubble">
                        <div>{message["content"]}</div>
                        <div class="message-time">🤖 AI助手</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 使用Streamlit原生组件显示来源信息
                if "sources" in message and message["sources"]:
                    with st.expander("📚 参考来源", expanded=False):
                        for i, source in enumerate(message["sources"][:3], 1):
                            st.markdown(f"""
                            **📄 {source.get('filename', '未知文件')}**
                            - **页码**: {source.get('page', '?')}
                            - **相关度**: {source.get('score', 'N/A')}
                            - **预览**: {source.get('preview', '')[:100]}...
                            """)
                            st.divider() if i < len(message["sources"][:3]) else None
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 聊天输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 立即添加用户消息并显示
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_history()
        
        # 立即重新运行以显示用户消息
        st.rerun()
    
    # 检查是否有正在处理的AI回复（通过session_state标记）
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_user_message = st.session_state.messages[-1]["content"]
        
        # 显示AI思考中的状态
        with st.container():
            st.markdown("""
            <div class="ai-message">
                <div class="ai-avatar">🤖</div>
                <div class="ai-bubble">
                    <div>🔍 正在检索相关文档并生成回答...</div>
                    <div class="message-time">🤖 AI助手</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 在后台执行RAG查询
        with st.spinner("🔍 正在检索相关文档并生成回答..."):
            # 获取当前知识库和设置
            knowledge_base = st.session_state.get("current_collection", "default")
            use_reranking = st.session_state.get("use_reranking", True)
            
            # 执行RAG查询
            result = query_rag(last_user_message, knowledge_base, use_reranking, include_context=False)
            
            # 构建AI回复消息
            ai_message = {
                "role": "assistant", 
                "content": result["answer"]
            }
            
            # 如果有来源信息，添加到消息中
            if "sources" in result and result["sources"]:
                ai_message["sources"] = result["sources"]
            
            # 添加AI回复到历史
            st.session_state.messages.append(ai_message)
            save_chat_history()
            
            # 重新运行以显示AI回复
            st.rerun()
    
    # 底部信息 - 放在输入框下面
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.messages:
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.caption(f"💬 总消息: {total_messages} | 👤 用户: {user_messages}")
        else:
            st.caption("💬 暂无消息")
    
    with col2:
        # 显示当前知识库和API状态
        kb_name = st.session_state.get("current_collection", "default")
        api_status = "🟢 已连接" if st.session_state.get("api_connected", False) else "🔴 未连接"
        st.caption(f"📚 知识库: {kb_name} | {api_status}")
    
    with col3:
        # 显示模型信息
        model_name = st.session_state.get("selected_model", "未选择")
        rerank_status = "✅" if st.session_state.get("use_reranking", True) else "❌"
        st.caption(f"🤖 模型: {model_name} | 重排序: {rerank_status}")

def render_chat_history():
    """渲染对话历史"""
    st.markdown("### 📚 对话历史")
    
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #666;">
            <div style="font-size: 3em; margin-bottom: 15px;">💭</div>
            <p>暂无对话历史</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # 历史统计
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        ai_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 总消息数", total_messages)
        with col2:
            st.metric("👤 用户消息", user_messages)
        with col3:
            st.metric("🤖 AI回复", ai_messages)

        # 导出和清除按钮
        col_export, col_clear = st.columns(2)
        with col_export:
            if st.button("📤 导出历史", use_container_width=True):
                export_data = {
                    "export_time": datetime.now().isoformat(),
                    "total_messages": total_messages,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="💾 下载文件",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col_clear:
            if st.button("🗑️ 清除历史", use_container_width=True):
                st.session_state.messages = []
                save_chat_history()
                st.success("✅ 对话历史已清除")
                st.rerun()

        # 显示历史消息预览
        st.markdown("#### 📝 消息预览")
        for i, message in enumerate(st.session_state.messages[-10:], 1):
            role_icon = "👤" if message["role"] == "user" else "🤖"
            role_name = "用户" if message["role"] == "user" else "AI助手"

            with st.expander(f"{role_icon} {role_name} #{i}"):
                st.write(message["content"])

import streamlit as st
import os
from dotenv import load_dotenv
from app import RAGSystem
from vector_store import VectorStore
from pathlib import Path
import re # Import regex

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="知识库问答系统",
    page_icon="📚",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_resource
def get_rag_system():
    """初始化并返回RAG系统实例"""
    print("正在初始化RAG系统...")
    return RAGSystem()

@st.cache_data(ttl=300)
def get_knowledge_bases(_rag_system: RAGSystem):
    """获取所有知识库及其文件列表"""
    print("正在获取知识库列表...")
    indices = _rag_system.retriever.get_all_indices()
    kb_details = {}
    if indices:
        for index in indices:
            display_name = index[4:] if index.startswith('rag_') else index
            try:
                files = _rag_system.vector_store.get_files_in_index(index)
                kb_details[display_name] = {
                    "index_name": index,
                    "files": files
                }
            except Exception as e:
                print(f"获取知识库 {index} 的文件列表时出错: {e}")
                kb_details[display_name] = {
                    "index_name": index,
                    "files": ["错误：无法获取文件列表"]
                }
    return kb_details

def display_image_with_caption(img_url: str, caption: str = None):
    """显示图片并处理可能的错误"""
    try:
        is_url = img_url.startswith(("http://", "https://"))
        if not is_url and not os.path.exists(img_url):
            st.warning(f"图片文件不存在: {Path(img_url).name}")
            return False
        
        # 使用列布局来限制图片大小
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Replace use_column_width with use_container_width
            st.image(img_url, caption=caption, use_container_width=True)
        return True
    except Exception as e:
        img_name = Path(img_url).name if not is_url else img_url
        st.warning(f"无法加载图片 {img_name}: {str(e)}")
        return False

def parse_and_render_llm_response(response_text: str):
    """解析LLM响应文本，分离主要内容、图片和引用，并渲染"""
    print("\n=== 原始LLM响应文本 ===")
    print(response_text)
    print("\n=== 开始解析和渲染 ===")
    
    # Separate main content and reference list (using --- as delimiter)
    # Use re.DOTALL so '.' matches newlines if the pattern spans lines
    # 找到最后一个 --- 分隔符的位置
    parts = response_text.rsplit('\n---\s*\n', maxsplit=1)
    main_content = parts[0]  # 最后一个分隔符之前的所有内容
    reference_section = parts[1] if len(parts) > 1 else ""  # 最后一个分隔符之后的内容
    
    print("\n=== 主要内容 ===")
    print(main_content)
    print("\n=== 引用部分 ===")
    print(reference_section)

    # --- Render Main Content ---
    # Find image markdown in main content: ![alt](url)
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')  # 修正图片匹配正则表达式
    images_in_content = list(image_pattern.finditer(main_content))
    
    print("\n=== 找到的图片 ===")
    for match in images_in_content:
        print(f"图片描述: {match.group(1)}")
        print(f"图片URL: {match.group(2)}")
    
    # Render content segment by segment, inserting images where markdown was
    last_end = 0
    for match in images_in_content:
        start, end = match.span()
        img_alt = match.group(1)
        img_url = match.group(2)
        
        # Render text before the image
        if start > last_end:
            text_segment = main_content[last_end:start]
            print(f"\n=== 渲染文本片段 ===\n{text_segment}")
            st.markdown(text_segment, unsafe_allow_html=True)
        
        # Render the image using the helper function
        print(f"\n=== 渲染图片 ===\n{img_url} (描述: {img_alt})")
        display_image_with_caption(img_url, caption=img_alt or "相关图片")
        
        last_end = end

    # Render text after the last image (or the whole text if no images)
    if last_end < len(main_content):
        text_segment = main_content[last_end:]
        print(f"\n=== 渲染剩余文本 ===\n{text_segment}")
        st.markdown(text_segment, unsafe_allow_html=True)
    elif not images_in_content and main_content:
        print(f"\n=== 渲染完整文本（无图片）===\n{main_content}")
        st.markdown(main_content, unsafe_allow_html=True)

    # --- Render References ---
    references = {}
    if reference_section:
        # Find reference lines: [number] text
        # Assumes references start at the beginning of a line after the separator
        ref_pattern = re.compile(r'^\[(\d+)\]\s*(.*)', re.MULTILINE)
        print("\n=== 解析引用 ===")
        for match in ref_pattern.finditer(reference_section.strip()):
            ref_num = int(match.group(1))
            ref_text = match.group(2).strip()
            print(f"引用 [{ref_num}]: {ref_text}")
            references[ref_num] = ref_text

    if references:
        st.write("--- 引用来源 ---")
        # Ensure references are sorted by number
        for i in sorted(references.keys()):
            # Render reference text as markdown to allow links within it
            st.markdown(f"[{i}] {references[i]}", unsafe_allow_html=True)
    
    print("\n=== 渲染完成 ===\n")

# --- Initialization ---
rag_system = get_rag_system()
knowledge_bases = get_knowledge_bases(rag_system)

# --- Sidebar ---
st.sidebar.title("知识库管理")
st.sidebar.divider()

if not knowledge_bases:
    st.sidebar.warning("当前没有知识库。请先创建。")
else:
    st.sidebar.subheader("现有知识库")
    for name, details in knowledge_bases.items():
        with st.sidebar.expander(f"📚 {name} ({len(details['files'])} 文件)"):
            if details["files"]:
                for i, file in enumerate(details["files"], 1):
                    # 只显示文件名，不显示完整路径
                    st.markdown(f"- {Path(file).name}")
            else:
                st.info("此知识库中暂无文件。")
    st.sidebar.divider()

# 创建新知识库
st.sidebar.subheader("创建新知识库")
new_kb_name = st.sidebar.text_input("知识库名称", key="new_kb_name")
new_kb_path = st.sidebar.text_input("文档路径 (文件或目录)", key="new_kb_path")
if st.sidebar.button("创建知识库", key="create_kb_button"):
    if new_kb_name and new_kb_path:
        normalized_path = os.path.normpath(new_kb_path)
        if os.path.exists(normalized_path):
            kb_name_clean = new_kb_name.lower().strip()
            index_name_internal = f"rag_{kb_name_clean}"
            
            # 检查知识库是否已存在
            current_indices = rag_system.retriever.get_all_indices()
            if index_name_internal in current_indices:
                st.sidebar.error(f"知识库 '{kb_name_clean}' 已存在！")
            else:
                try:
                    with st.spinner(f"正在处理和索引文档到 '{kb_name_clean}'..."):
                        rag_system.process_documents(new_kb_path, kb_name_clean)
                    st.sidebar.success(f"知识库 '{kb_name_clean}' 创建成功！")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"创建时出错: {str(e)}")
        else:
            st.sidebar.error(f"路径不存在: {normalized_path}")
    else:
        st.sidebar.warning("请输入知识库名称和文档路径。")

st.sidebar.divider()

# 向已有知识库添加文档
st.sidebar.subheader("向已有知识库添加文档")
if knowledge_bases:
    kb_names_list = list(knowledge_bases.keys())
    selected_kb_to_add = st.sidebar.selectbox("选择知识库", kb_names_list, key="add_doc_kb_select")
    add_doc_path = st.sidebar.text_input("要添加的文档路径", key="add_doc_path")
    if st.sidebar.button("添加文档", key="add_doc_button"):
        if selected_kb_to_add and add_doc_path:
            normalized_path = os.path.normpath(add_doc_path)
            if os.path.exists(normalized_path):
                try:
                    with st.spinner(f"正在添加文档到 '{selected_kb_to_add}'..."):
                        rag_system.process_documents(add_doc_path, selected_kb_to_add.lower().strip())
                    st.sidebar.success(f"文档已添加到 '{selected_kb_to_add}'！")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"添加文档时出错: {str(e)}")
            else:
                st.sidebar.error(f"路径不存在: {normalized_path}")
        else:
            st.sidebar.warning("请选择知识库并输入文档路径。")
else:
    st.sidebar.info("没有可用的知识库来添加文档。")

# --- Main Chat Interface ---
st.title("💬 知识库问答")
st.divider()

if not knowledge_bases:
    st.info("请先在左侧侧边栏创建或选择一个知识库。")
else:
    # 选择知识库进行对话
    kb_names_list = list(knowledge_bases.keys())
    if 'selected_kb_chat' not in st.session_state or st.session_state.selected_kb_chat not in kb_names_list:
        st.session_state.selected_kb_chat = kb_names_list[0]

    st.session_state.selected_kb_chat = st.selectbox(
        "选择对话知识库",
        kb_names_list,
        index=kb_names_list.index(st.session_state.selected_kb_chat),
        key="chat_kb_selector"
    )
    st.info(f"当前对话知识库: **{st.session_state.selected_kb_chat}** (检索时将查询所有可用知识库)")

    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                # Parse and render the stored raw response
                # Handle potential errors during rendering as well
                try:
                    parse_and_render_llm_response(message["content"])
                except Exception as render_error:
                    st.error(f"渲染历史消息时出错: {render_error}")
                    # Display raw content as fallback
                    st.markdown(message["content"])

    # 接收用户输入
    if prompt := st.chat_input(f"向知识库 '{st.session_state.selected_kb_chat}' 提问..."):
        # 添加用户消息到聊天记录
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成并显示回答
        with st.chat_message("assistant"):
            # Use a container for the whole response rendering
            response_container = st.container()
            try:
                with st.spinner("思考中..."):
                    # 获取回答 (原始文本)
                    response_text, reranked_docs = rag_system.query(prompt)
                    
                    # Parse and render the response within the container
                    with response_container:
                        parse_and_render_llm_response(response_text)
                    
                    # 将原始回答添加到聊天记录
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text, # Store only the raw response text
                    })
                    
            except Exception as e:
                error_message = f"处理您的问题时出错: {str(e)}"
                with response_container:
                    st.error(error_message)
                # Store the error message as the content
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message # Store error as content
                }) 
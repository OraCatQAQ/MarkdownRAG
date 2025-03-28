from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    TextLoader
)
import os
import requests
import base64
from PIL import Image
import io
import subprocess
import json
import re
import concurrent.futures
from pathlib import Path

# Helper function to normalize paths
def normalize_path(path_str: str) -> str:
    """Converts a path string to an absolute path with forward slashes."""
    return Path(path_str).resolve().as_posix()

class DocumentLoader:
    """通用文档加载器"""
    def __init__(self, file_path: str):
        self.file_path = normalize_path(file_path)
        self.extension = os.path.splitext(self.file_path)[1].lower()
        self.api_key = os.getenv("API_KEY")
        self.api_base = os.getenv("BASE_URL")
        
    def process_image(self, image_path: str, context: str = None) -> str:
        """使用 SiliconFlow VLM 模型处理图片，可选择性地提供上下文"""
        try:
            # 读取图片并转换为base64
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # 准备提示词，如果有上下文则包含在内
            prompt = """请分析这张图片是否包含有意义的信息。
            
如果图片是装饰性的、无实质内容的配图（如分隔线、背景图、装饰性插图等），请直接回复"None"。
            
如果图片包含有意义的信息（如图表、数据可视化、流程图、实质性内容的照片等），请详细描述这张图片的内容，包括主要对象、场景、活动、颜色、布局等关键信息。"""

            if context:
                prompt = f"""这张图片出现在以下上下文中：

{context}

请根据上下文分析这张图片是否包含有意义的信息。

如果图片只是装饰性的、无实质内容的配图（如分隔线、背景图、装饰性插图等），请直接回复"None"。

如果图片包含有意义的信息（如图表、数据可视化、流程图、实质性内容的照片等），请描述这张图片的内容"""
            
            # 调用 SiliconFlow API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json={
                    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"图片处理API调用失败: {response.text}")
                
            description = response.json()["choices"][0]["message"]["content"]
            
            # 检查是否返回None（忽略大小写和空格）
            if description.strip().lower() == "None" or len(description.strip()) < 10:
                return None
                
            return description
            
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return "图片处理失败"
    
    def process_pdf_with_magic(self, pdf_path: str) -> str:
        """使用magic-pdf处理PDF文件"""
        try:
            # 获取文件名(不含扩展名)作为处理目录
            file_hash = Path(pdf_path).stem
            output_dir = normalize_path(f"./output/{file_hash}/auto")
            pdf_abs_path = pdf_path

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 执行magic-pdf命令
            command = f"magic-pdf -p \"{pdf_abs_path}\" -o ./output"
            subprocess.run(command, shell=True, check=True, cwd=Path.cwd())
            
            # 读取生成的markdown文件
            markdown_path = os.path.join(output_dir, f"{file_hash}.md")
            if not os.path.exists(markdown_path):
                raise Exception("Markdown文件未生成")
                
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 将相对路径的图片引用转换为绝对路径
            img_dir_abs = normalize_path(os.path.join(output_dir, "images"))
            content = re.sub(r'\]\(images/', f']({img_dir_abs}/', content)
            
            return content
                
        except Exception as e:
            print(f"PDF处理失败: {str(e)}")
            raise
    
    def process_markdown(self, content: str) -> List[Dict]:
        """处理Markdown内容，提取标题层级和图片信息"""
        chunks = []
        current_headers = []  # 用于跟踪标题层级
        current_content = []
        image_references = []  # 收集所有图片引用
        
        # 获取markdown文件所在目录，用于解析相对路径
        base_dir = Path(self.file_path).parent
        
        for line in content.split('\n'):
            # 检查是否是标题
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # 如果有累积的内容，先处理
                if current_content:
                    chunks.append({
                        'content': '\n'.join(current_content),
                        'headers': current_headers.copy(),
                        'img_url': None
                    })
                    current_content = []
                
                # 更新当前标题层级
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_headers = current_headers[:level-1] + [title]
                current_content.append(line)
                
            # 检查是否是图片
            elif line.startswith('!['):
                img_match = re.search(r'!\[.*?\]\((.*?)\)', line)
                if img_match:
                    img_path_rel = img_match.group(1)
                    
                    # 处理图片路径
                    try:
                        # Check if path is already absolute (might happen if PDF processor output absolute paths)
                        if Path(img_path_rel).is_absolute():
                            img_path_abs = normalize_path(img_path_rel)
                        else:
                            # Resolve relative to the markdown file's directory
                            img_path_abs = normalize_path(base_dir / img_path_rel)
                    except Exception as path_e:
                        print(f"警告：处理图片路径时出错 '{img_path_rel}': {path_e}")
                        current_content.append(line)
                        continue
                    
                    # 验证图片文件是否存在
                    if not os.path.exists(img_path_abs):
                        print(f"警告：图片文件不存在: {img_path_abs}")
                        current_content.append(line)
                        continue
                    
                    # 如果有累积的内容，先处理
                    if current_content:
                        chunks.append({
                            'content': '\n'.join(current_content),
                            'headers': current_headers.copy(),
                            'img_url': None
                        })
                        current_content = []
                    
                    # 收集图片信息，稍后处理
                    image_references.append({
                        'img_path': img_path_abs,
                        'position': len(chunks),
                        'headers': current_headers.copy()
                    })
                    
                    # 添加一个占位符块
                    chunks.append({
                        'content': f"[图片占位符]",
                        'headers': current_headers.copy(),
                        'img_url': img_path_abs
                    })
                else:
                    current_content.append(line)
            else:
                current_content.append(line)
        
        # 处理最后剩余的内容
        if current_content:
            chunks.append({
                'content': '\n'.join(current_content),
                'headers': current_headers.copy(),
                'img_url': None
            })
        
        # 并发处理所有图片
        if image_references:
            self._process_images_concurrently(chunks, image_references)
            
        return chunks
    
    def _process_images_concurrently(self, chunks: List[Dict], image_references: List[Dict]):
        """并发处理所有图片，包含上下文信息，并过滤无意义的图片"""
        def process_single_image(ref):
            try:
                # 获取图片的上下文
                context = self._get_image_context(chunks, ref)
                img_description = self.process_image(ref['img_path'], context)
                
                # 如果VLM返回None，表示图片无意义
                if img_description is None:
                    return {
                        'position': ref['position'],
                        'description': None,
                        'img_path': ref['img_path'],
                        'meaningful': False
                    }
                
                return {
                    'position': ref['position'],
                    'description': img_description,
                    'img_path': ref['img_path'],
                    'meaningful': True
                }
            except Exception as e:
                print(f"处理图片时出错 {ref['img_path']}: {str(e)}")
                return {
                    'position': ref['position'],
                    'description': "图片处理失败",
                    'img_path': ref['img_path'],
                    'meaningful': True  # 出错时默认保留
                }
        
        # 使用线程池并发处理图片
        positions_to_remove = []  # 记录需要删除的位置
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_img = {executor.submit(process_single_image, ref): ref for ref in image_references}
            for future in concurrent.futures.as_completed(future_to_img):
                result = future.result()
                if result:
                    position = result['position']
                    
                    # 如果图片无意义，记录需要删除的位置
                    if not result['meaningful']:
                        positions_to_remove.append(position)
                    # 否则更新对应位置的块
                    elif 0 <= position < len(chunks):
                        chunks[position]['content'] = f"图片描述：{result['description']}"
                        chunks[position]['img_url'] = result['img_path']
                        chunks[position]['headers'] = "img"
        
        # 从后向前删除无意义的图片块，避免索引变化
        for position in sorted(positions_to_remove, reverse=True):
            if 0 <= position < len(chunks):
                del chunks[position]
    
    def _get_image_context(self, chunks: List[Dict], image_ref: Dict) -> str:
        """获取图片的上下文信息"""
        context_parts = []
        
        # 添加标题层级作为上下文
        if image_ref['headers']:
            context_parts.append(" > ".join(image_ref['headers']))
        
        # 获取图片前后的文本块作为上下文
        position = image_ref['position']
        
        # 获取前一个块的内容（如果存在）
        if position > 0 and 'content' in chunks[position-1]:
            prev_content = chunks[position-1]['content']
            # 只取最后500个字符作为上下文
            if len(prev_content) > 500:
                prev_content = "..." + prev_content[-500:]
            context_parts.append(f"图片前文本：{prev_content}")
        
        # 获取后一个块的内容（如果存在）
        if position < len(chunks) - 1 and 'content' in chunks[position+1]:
            next_content = chunks[position+1]['content']
            # 只取前500个字符作为上下文
            if len(next_content) > 500:
                next_content = next_content[:500] + "..."
            context_parts.append(f"图片后文本：{next_content}")
        
        return "\n\n".join(context_parts)
    
    def load(self):
        try:
            if self.extension == '.pdf':
                # 使用magic-pdf处理PDF
                content = self.process_pdf_with_magic(self.file_path)
                # 处理生成的markdown内容
                chunks = self.process_markdown(content)
                
                # 转换为Document格式
                from langchain.schema import Document
                docs = []
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk['content'],
                        metadata={
                            'source': self.file_path,
                            'chunk_header': ' > '.join(chunk['headers']) if chunk['headers'] else '',
                            'img_url': chunk['img_url'] if chunk['img_url'] else ''
                        }
                    )
                    docs.append(doc)
                return docs
                
            elif self.extension == '.md':
                # 直接读取markdown文件
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = self.process_markdown(content)
                
                # 转换为Document格式
                from langchain.schema import Document
                docs = []
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk['content'],
                        metadata={
                            'source': self.file_path,
                            'chunk_header': ' > '.join(chunk['headers']) if chunk['headers'] else '',
                            'img_url': chunk['img_url'] if chunk['img_url'] else ''
                        }
                    )
                    docs.append(doc)
                return docs
                
            elif self.extension == '.txt':
                loader = TextLoader(self.file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = self.file_path
                return docs
            elif self.extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                # 处理图片
                description = self.process_image(self.file_path)
                # 创建一个包含图片描述的文档
                from langchain.schema import Document
                doc = Document(
                    page_content=description if description else "无法处理的图片",
                    metadata={
                        'source': self.file_path,
                        'img_url': self.file_path
                    }
                )
                return [doc]
            else:
                raise ValueError(f"不支持的文件格式: {self.extension}")
                
        except UnicodeDecodeError:
            # 如果 utf-8 失败，尝试 gbk
            if self.extension in ['.md', '.txt']:
                try:
                    loader = TextLoader(self.file_path, encoding='gbk')
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = self.file_path
                    return docs
                except Exception as e_gbk:
                    print(f"尝试 GBK 解码失败: {e_gbk}")
                    raise
            raise

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def get_index_name(self, path: str) -> str:
        """根据文件路径生成索引名称"""
        if os.path.isdir(path):
            # 如果是目录，使用目录名
            return f"rag_{os.path.basename(path).lower()}"
        else:
            # 如果是文件，使用文件名（不含扩展名）
            return f"rag_{os.path.splitext(os.path.basename(path))[0].lower()}"
        
    def process(self, path: str) -> List[Dict]:
        """
        加载并处理文档，支持目录或单个文件
        返回：处理后的文档列表
        """
        normalized_input_path = normalize_path(path)

        all_loaded_docs = []
        if os.path.isdir(normalized_input_path):
            for root, _, files in os.walk(normalized_input_path):
                for file in files:
                    file_path_abs = normalize_path(os.path.join(root, file))
                    try:
                        loader = DocumentLoader(file_path_abs)
                        docs = loader.load()
                        # 添加文件名到metadata
                        file_name = Path(file_path_abs).name
                        for doc in docs:
                            doc.metadata['file_name'] = file_name
                            if 'source' not in doc.metadata or not doc.metadata['source']:
                                doc.metadata['source'] = file_path_abs
                        all_loaded_docs.extend(docs)
                    except Exception as e:
                        print(f"警告：加载文件 {file_path_abs} 时出错: {str(e)}")
                        continue
        else:
            try:
                loader = DocumentLoader(normalized_input_path)
                docs = loader.load()
                # 添加文件名到metadata
                file_name = Path(normalized_input_path).name
                for doc in docs:
                    doc.metadata['file_name'] = file_name
                    if 'source' not in doc.metadata or not doc.metadata['source']:
                        doc.metadata['source'] = normalized_input_path
                all_loaded_docs.extend(docs)
            except Exception as e:
                print(f"加载文件时出错: {str(e)}")
                raise
        
        # 分块
        chunks = self.text_splitter.split_documents(all_loaded_docs)
        
        # 处理成统一格式
        processed_docs = []
        for i, chunk in enumerate(chunks):
            processed_docs.append({
                'id': f'doc_{i}',
                'content': chunk.page_content,
                'metadata': {
                    'file_name': chunk.metadata.get('file_name', Path(chunk.metadata.get('source', '未知文件')).name),
                    'source': chunk.metadata.get('source', ''),
                    'chunk_header': chunk.metadata.get('chunk_header', ''),
                    'img_url': chunk.metadata.get('img_url', '')
                }
            })
            
        return processed_docs
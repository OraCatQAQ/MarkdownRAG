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

class DocumentLoader:
    """通用文档加载器"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extension = os.path.splitext(file_path)[1].lower()
        self.api_key = os.getenv("API_KEY")
        self.api_base = os.getenv("BASE_URL")
        
    def process_image(self, image_path: str) -> str:
        """使用 SiliconFlow VLM 模型处理图片"""
        try:
            # 读取图片并转换为base64
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
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
                                    "text": "请详细描述这张图片的内容，包括主要对象、场景、活动、颜色、布局等关键信息。"
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
            return description
            
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return "图片处理失败"
    
    def load(self):
        try:
            if self.extension == '.md':
                loader = UnstructuredMarkdownLoader(self.file_path, encoding='utf-8')
                return loader.load()
            elif self.extension == '.pdf':
                loader = PyPDFLoader(self.file_path)
                return loader.load()
            elif self.extension == '.txt':
                loader = TextLoader(self.file_path, encoding='utf-8')
                return loader.load()
            elif self.extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                # 处理图片
                description = self.process_image(self.file_path)
                # 创建一个包含图片描述的文档
                from langchain.schema import Document
                doc = Document(
                    page_content=description,
                    metadata={
                        'source': self.file_path,
                        'img_url': os.path.abspath(self.file_path)  # 存储图片的绝对路径
                    }
                )
                return [doc]
            else:
                raise ValueError(f"不支持的文件格式: {self.extension}")
                
        except UnicodeDecodeError:
            # 如果 utf-8 失败，尝试 gbk
            if self.extension in ['.md', '.txt']:
                loader = TextLoader(self.file_path, encoding='gbk')
                return loader.load()
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
        if os.path.isdir(path):
            documents = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        loader = DocumentLoader(file_path)
                        docs = loader.load()
                        # 添加文件名到metadata
                        for doc in docs:
                            doc.metadata['file_name'] = os.path.basename(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"警告：加载文件 {file_path} 时出错: {str(e)}")
                        continue
        else:
            try:
                loader = DocumentLoader(path)
                documents = loader.load()
                # 添加文件名到metadata
                file_name = os.path.basename(path)
                for doc in documents:
                    doc.metadata['file_name'] = file_name
            except Exception as e:
                print(f"加载文件时出错: {str(e)}")
                raise
        
        # 分块
        chunks = self.text_splitter.split_documents(documents)
        
        # 处理成统一格式
        processed_docs = []
        for i, chunk in enumerate(chunks):
            processed_docs.append({
                'id': f'doc_{i}',
                'content': chunk.page_content,
                'metadata': chunk.metadata
            })
            
        return processed_docs
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    TextLoader
)
import os

class DocumentLoader:
    """通用文档加载器"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extension = os.path.splitext(file_path)[1].lower()
        
    def load(self):
        try:
            if self.extension == '.md':
                loader = UnstructuredMarkdownLoader(self.file_path, encoding='utf-8')
            elif self.extension == '.pdf':
                loader = PyPDFLoader(self.file_path)
            elif self.extension == '.txt':
                loader = TextLoader(self.file_path, encoding='utf-8')
            else:
                raise ValueError(f"不支持的文件格式: {self.extension}")
                
            return loader.load()
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
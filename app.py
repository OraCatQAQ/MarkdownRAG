from typing import List, Dict
import os
import argparse
from document_processor import DocumentProcessor
from vector_store import VectorStore
from retriever import Retriever
from reranker import Reranker
from generator import Generator

class RAGSystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()
    
    def show_indexed_files(self) -> List[str]:
        """显示已索引的文件"""
        indices = self.retriever.get_all_indices()
        if not indices:
            print("\n当前没有知识库。")
            return []
        
        print("\n现有知识库：")
        print("=" * 50)
        for i, index in enumerate(indices, 1):
            # 移除 'rag_' 前缀
            display_name = index[4:] if index.startswith('rag_') else index
            # 获取该知识库中的文件列表
            files = self.vector_store.get_files_in_index(index)
            print(f"\n{i}. 知识库：{display_name}")
            if files:
                print("   包含以下文件：")
                for j, file in enumerate(files, 1):
                    print(f"      {j}) {file}")
            else:
                print("   暂无文件")
            print("-" * 50)
        return indices
    
    def process_documents(self, documents_path: str, index_name: str) -> None:
        """处理并索引文档到指定知识库"""
        print(f"开始处理文档: {documents_path}")
        # 处理文档
        processed_docs = self.doc_processor.process(documents_path)
        print(f"文档处理完成，共处理 {len(processed_docs)} 个文档片段")
        
        # 存储到向量数据库
        print(f"正在将文档存入知识库（{index_name}）...")
        self.vector_store.store(processed_docs, f"rag_{index_name}")
        print("文档存储完成！")
    
    def query(self, query: str) -> str:
        """处理用户查询"""
        print("\n正在检索相关文档...")
        # 检索相关文档
        retrieved_docs, index_name = self.retriever.retrieve(query)
        
        print("正在重排序文档...")
        # 重排序
        reranked_docs = self.reranker.rerank(query, retrieved_docs, index_name)
        
        print("正在生成回答...\n")
        # 生成回答
        response = self.generator.generate(query, reranked_docs)
        return response

def main():
    # 初始化RAG系统
    rag_system = RAGSystem()
    
    try:
        while True:
            # 显示已索引的文件
            indices = rag_system.show_indexed_files()
            
            # 询问用户操作
            print("\n请选择操作：")
            print("1. 直接开始问答")
            print("2. 创建新的知识库")
            print("3. 向已有知识库添加文档")
            print("4. 退出程序")
            
            choice = input("\n请输入选项（1-4）: ").strip()
            
            if choice == "1":
                if not indices:
                    print("错误：当前没有知识库，请先创建知识库并添加文档！")
                    continue
                    
                # 进入问答循环
                print("\n开始问答（输入 'q' 或 'quit' 返回主菜单）")
                while True:
                    query = input("\n请输入您的问题: ").strip()
                    
                    if query.lower() in ['q', 'quit', 'exit']:
                        print("返回主菜单...")
                        break
                    
                    if not query:
                        print("问题不能为空，请重新输入！")
                        continue
                    
                    try:
                        # 获取回答
                        response = rag_system.query(query)
                        print("\n回答：")
                        print(response)
                    except Exception as e:
                        print(f"\n生成回答时出错：{str(e)}")
                        print("请重试或联系管理员。")
            
            elif choice == "2":
                # 创建新知识库
                index_name = input("\n请输入知识库名称: ").strip().lower()
                if not index_name:
                    print("错误：知识库名称不能为空！")
                    continue
                
                if f"rag_{index_name}" in indices:
                    print(f"错误：知识库 '{index_name}' 已存在！")
                    continue
                
                # 添加文档
                docs_path = input("\n请输入要添加的文档路径: ").strip()
                docs_path = os.path.normpath(docs_path)
                
                if not os.path.exists(docs_path):
                    print(f"错误：路径 '{docs_path}' 不存在！")
                    continue
                
                try:
                    rag_system.process_documents(docs_path, index_name)
                    print("知识库创建成功！")
                except Exception as e:
                    print(f"创建知识库时出错：{str(e)}")
                    print("请检查文档格式或联系管理员。")
            
            elif choice == "3":
                # 向已有知识库添加文档
                if not indices:
                    print("错误：当前没有知识库，请先创建知识库！")
                    continue
                
                print("\n请选择要添加文档的知识库：")
                for i, index in enumerate(indices, 1):
                    display_name = index[4:] if index.startswith('rag_') else index
                    print(f"{i}. {display_name}")
                
                try:
                    idx = int(input("\n请输入知识库编号: ").strip()) - 1
                    if not 0 <= idx < len(indices):
                        print("无效的知识库编号！")
                        continue
                    
                    # 添加文档
                    docs_path = input("\n请输入要添加的文档路径: ").strip()
                    docs_path = os.path.normpath(docs_path)
                    
                    if not os.path.exists(docs_path):
                        print(f"错误：路径 '{docs_path}' 不存在！")
                        continue
                    
                    selected_index = indices[idx][4:] if indices[idx].startswith('rag_') else indices[idx]
                    rag_system.process_documents(docs_path, selected_index)
                    print("文档添加成功！")
                except ValueError:
                    print("请输入有效的数字！")
                except Exception as e:
                    print(f"添加文档时出错：{str(e)}")
                    print("请检查文档格式或联系管理员。")
            
            elif choice == "4":
                print("\n感谢使用！再见！")
                break
            
            else:
                print("\n无效的选项，请重新选择！")
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。感谢使用！")
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")

if __name__ == "__main__":
    main()

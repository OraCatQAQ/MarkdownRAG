from typing import List, Dict, Tuple
import requests
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

class Retriever:
    def __init__(self):
        # 使用与 vector_store.py 相同的 ES 配置
        self.es = Elasticsearch(
            "https://localhost:9200",  # 注意是 https
            basic_auth=("elastic", os.getenv("PASSWORD")),  # 使用相同的密码
            verify_certs=False  # 开发环境可以禁用证书验证
        )
        self.api_key = os.getenv("API_KEY")
        self.api_base = os.getenv("BASE_URL")
        
    def get_embedding(self, text: str) -> List[float]:
        """调用SiliconFlow的embedding API获取向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json={
                "model": "BAAI/bge-m3",
                "input": text
            }
        )
        
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(f"Error getting embedding: {response.text}")
    
    def get_all_indices(self) -> List[str]:
        """获取所有 RAG 相关的索引"""
        indices = self.es.indices.get_alias().keys()
        return [idx for idx in indices if idx.startswith('rag_')]
        
    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[Dict], str]:
        """混合检索：结合 BM25 和向量检索"""
        # 获取所有 RAG 索引
        indices = self.get_all_indices()
        if not indices:
            raise Exception("没有找到可用的文档索引！")
            
        # 计算查询向量
        query_vector = self.get_embedding(query)
        
        # 在所有索引中搜索
        all_results = []
        for index in indices:
            # 构建混合查询
            script_query = {
                "script_score": {
                    "query": {
                        "match": {
                            "content": query  # BM25
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
            
            # 执行检索
            response = self.es.search(
                index=index,
                body={
                    "query": script_query,
                    "size": top_k
                }
            )
            
            # 处理结果
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'content': hit['_source']['content'],
                    'score': hit['_score'],
                    'metadata': hit['_source']['metadata'],
                    'index': index
                }
                all_results.append(result)
        
        # 按分数排序并选择最相关的文档
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:top_k]
        
        # 如果有结果，返回最相关文档所在的索引
        if top_results:
            most_relevant_index = top_results[0]['index']
        else:
            most_relevant_index = indices[0]  # 如果没有结果，返回第一个索引
            
        return top_results, most_relevant_index 
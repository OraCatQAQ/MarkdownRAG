from typing import List, Dict
import requests
from dotenv import load_dotenv
import os

load_dotenv()

class Reranker:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_base = os.getenv("BASE_URL")
        
    def rerank(self, query: str, documents: List[Dict], index_name: str, top_k: int = 5) -> List[Dict]:
        """使用SiliconFlow的rerank API重排序文档"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 准备文档列表
        docs = [doc['content'] for doc in documents]
        
        response = requests.post(
            f"{self.api_base}/rerank",
            headers=headers,
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": docs,
                "top_n": top_k
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error in reranking: {response.text}")
            
        # 处理结果
        results = response.json()["results"]
        reranked_docs = []
        
        for result in results:
            doc_index = result["index"]
            original_doc = documents[doc_index].copy()
            original_doc['rerank_score'] = result["relevance_score"]
            original_doc['index_name'] = index_name  
            reranked_docs.append(original_doc)
            
        return reranked_docs 
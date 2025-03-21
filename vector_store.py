from typing import List, Dict
import requests
import numpy as np
from elasticsearch import Elasticsearch
import urllib3
from dotenv import load_dotenv
import os

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class VectorStore:
    def __init__(self):
        # ES 8.x 的连接配置
        self.es = Elasticsearch(
            "https://localhost:9200",
            basic_auth=("elastic", os.getenv("PASSWORD")),
            verify_certs=False,
            request_timeout=30,
            # 忽略系统索引警告
            headers={"accept": "application/vnd.elasticsearch+json; compatible-with=8"},
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
    
    def store(self, documents: List[Dict], index_name: str) -> None:
        """将文档存储到 Elasticsearch"""
        # 创建索引（如果不存在）
        if not self.es.indices.exists(index=index_name):
            self.create_index(index_name)
        
        # 获取当前索引中的文档数量
        try:
            response = self.es.count(index=index_name)
            last_id = response['count'] - 1  # 文档数量减1作为最后的ID
            if last_id < 0:
                last_id = -1
        except Exception as e:
            print(f"获取文档数量时出错，假设为-1: {str(e)}")
            last_id = -1
        
        # 批量索引文档
        bulk_data = []
        for i, doc in enumerate(documents, start=last_id + 1):
            # 获取文档向量
            vector = self.get_embedding(doc['content'])
            
            # 准备索引数据
            bulk_data.append({
                "index": {
                    "_index": index_name,
                    "_id": f"doc_{i}"
                }
            })
            
            # 构建文档数据，包含新的img_url字段
            doc_data = {
                "content": doc['content'],
                "vector": vector,
                "metadata": {
                    "file_name": doc['metadata'].get('file_name', '未知文件'),
                    "source": doc['metadata'].get('source', ''),
                    "page": doc['metadata'].get('page', ''),
                    "img_url": doc['metadata'].get('img_url', '')  # 添加img_url字段
                }
            }
            bulk_data.append(doc_data)
            
        # 批量写入
        if bulk_data:
            response = self.es.bulk(operations=bulk_data, refresh=True)
            if response.get('errors'):
                print("批量写入时出现错误：", response)
    
    def get_files_in_index(self, index_name: str) -> List[str]:
        """获取索引中的所有文件名"""
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "unique_files": {
                            "terms": {
                                "field": "metadata.file_name",
                                "size": 1000
                            }
                        }
                    }
                }
            )
            
            files = [bucket['key'] for bucket in response['aggregations']['unique_files']['buckets']]
            return sorted(files)
        except Exception as e:
            print(f"获取文件列表时出错: {str(e)}")
            return []

    def create_index(self, index_name: str):
        """创建 Elasticsearch 索引"""
        settings = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": 1024
                    },
                    "metadata": {
                        "properties": {
                            "file_name": {
                                "type": "keyword",
                                "ignore_above": 256
                            },
                            "source": {
                                "type": "keyword"
                            },
                            "page": {
                                "type": "keyword"
                            },
                            "img_url": {  # 新增图片URL字段
                                "type": "keyword",
                                "ignore_above": 2048
                            }
                        }
                    }
                }
            }
        }
        
        # 如果索引已存在，先删除
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        
        self.es.indices.create(index=index_name, body=settings) 
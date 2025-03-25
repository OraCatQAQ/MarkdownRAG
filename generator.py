from typing import List, Dict
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_base = os.getenv("BASE_URL")
        
    def generate(self, query: str, context_docs: List[Dict]) -> str:
        """使用SiliconFlow的chat API生成回答"""
        # 构建带有引用标记的上下文
        context_with_refs = []
        
        for i, doc in enumerate(context_docs, 1):
            metadata = doc['metadata']
            file_name = metadata.get('file_name', '未知文件')
            chunk_header = metadata.get('chunk_header', '')
            img_url = metadata.get('img_url', '')
            source = metadata.get('source', '')
            
            # 构建引用标记
            ref_header = f"[{i}] "
            if chunk_header:
                ref_header += f"【{chunk_header}】"
            
            # 构建文档内容
            content = doc['content']
            
            # 如果是图片描述，特殊处理
            if content.startswith("图片描述：") and img_url:
                context_with_refs.append(f"{ref_header}{content}\nimg_url: {img_url}")
            else:
                # 普通文本内容
                file = f"file_name: {file_name}"
                source = f"source: {source}"
                header = f"header: {chunk_header}"
                context_with_refs.append(f"{ref_header}{content}\n{file}\n{source}\n{header}")
        
        context = "\n\n".join(context_with_refs)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """你是一个有帮助的助手。请基于提供的参考内容回答问题，使用markdown格式。
1. 如果回答中引用参考内容，请在相关内容后用方括号标注来源编号，例如：[1]、[2]
2. 如果参考内容中包含图片描述，在适当位置用markdown格式通过img_url给出图片即可
3. 最后给出引用列表: 将引用内容总结并给出来源和位置

参考格式：
---
[1] 一句话概括 [file_name](source) header
"""

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json={
                "model": "deepseek-ai/DeepSeek-V3",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
参考内容：
{context}

问题：{query}

请按照要求回答问题。
"""}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error in generation: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"] 
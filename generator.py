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
            # 直接使用文件名作为来源
            file_name = doc['metadata'].get('file_name', '未知文件')
            context_with_refs.append(f"[{i}] {doc['content']}\n来源：{file_name}")
        
        context = "\n\n".join(context_with_refs)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """你是一个有帮助的助手。请基于提供的参考内容回答问题。
1. 如果从参考内容中找到答案，请在相关内容后用方括号标注来源编号，例如：[1]、[2]
2. 如果内容来自多个来源，请标注所有相关来源
3. 如果无法从参考内容中得到答案，请明确说明
4. 回答要简洁清晰，避免重复引用
5. 在回答的最后，列出所有引用的文件名称"""

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

请按照要求回答问题，包括引用标注和来源列表。
"""}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error in generation: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"] 
# MarkdownRag

将带有图片的复杂markdown/pdf装入知识库，支持关于图片内容的知识库问答和图片输出
## 功能特点

- 🚀 支持多种文档格式：PDF、Markdown、TXT、图片（PNG/JPG/JPEG/GIF/BMP）
- 📚 支持创建多个知识库，方便文档管理
- 🔍 智能文档解析，自动提取文本内容和图片
- 🖼️ 支持图片描述和展示
- 📝 生成引用来源
- 🌐 友好的 Web 界面

## 快速开始

### 1. 环境要求

- Python 3.8+
- Elasticsearch 8.x
- SiliconFlow API 密钥（用于向量嵌入和问答生成）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 Elasticsearch

1. 下载并安装 [Elasticsearch 8.x](https://www.elastic.co/downloads/elasticsearch)

2. 启动 Elasticsearch 服务：
   ```bash
   # Windows
   .\elasticsearch.bat
   ```

3. 记录 Elasticsearch 的密码

### 4. 环境配置

创建 `.env` 文件并配置以下环境变量：

```env
# Elasticsearch 配置
PASSWORD=your_elasticsearch_password

# SiliconFlow API 配置
API_KEY=your_siliconflow_api_key
BASE_URL=https://api.siliconflow.com/v1
```

### 5. 启动应用

```bash
streamlit run ui.py
```

访问 http://localhost:8501 即可使用系统。

## 使用示例
![](https://tuchuang-1330806039.cos.ap-beijing.myqcloud.com/20250328184705883.png)
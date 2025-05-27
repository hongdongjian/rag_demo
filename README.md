# RAG Demo

```angular2html
# data目录，.txt文件会测试数据
# 将测试数据加载到向量数据库中
python3 load.py

# 大模型用的是豆包，ollama部署的deep-seek不支持function calling
# 可以修改config.py里的模型配置

# 启动MCP_Server，工具支持：保存内容到本地
python3 mcp_server.py

# 启动RAG服务
python3 mcp_client.py
```
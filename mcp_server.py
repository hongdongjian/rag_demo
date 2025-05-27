# mcp_server.py
from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("DemoServer")

@mcp.tool()
def save(content: str) -> str:
    """保存内容到本地"""
    with open("./data/test.log", 'w') as f:
        f.write(content)
    return f"Content saved to test.log"

if __name__ == "__main__":
    # 本地通信模式启动
    # mcp.run(transport='stdio')
    # 或启动HTTP服务：
    mcp.run(transport='sse')
import asyncio
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

from config import LLM_MODEL, LLM_URL, LLM_API_KEY, MCP_SERVER_SCRIPT, MCP_SERVER_URL
from retriever import HybridRetriever

# 加载环境变量
load_dotenv()


class MCPClient:
    def __init__(self):
        """初始化MCP客户端"""
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            base_url=LLM_URL,
            api_key=LLM_API_KEY
        )
        # 创建检索器实例（只创建一次）
        self.retriever = HybridRetriever()
        # 添加对话记忆
        self.conversation_history = []

    async def connect_to_stdio_server(self, server_script_path: str = MCP_SERVER_SCRIPT):
        """连接到标准输入输出的MCP服务器

        Args:
            server_script_path: 服务器脚本路径（.py或.js文件）
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是.py或.js文件")

        command = "python3" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()
            print("成功连接到标准输入输出服务器")
        except Exception as e:
            print(f"连接到标准输入输出服务器失败: {e}")
            raise

    async def connect_to_sse_server(self, server_url: str = MCP_SERVER_URL):
        """连接到MCP服务器的SSE端点

        Args:
            server_url: 服务器URL，默认为http://127.0.0.1:8000/sse
        """
        try:
            # 使用SSE客户端连接到服务器
            sse_transport = await self.exit_stack.enter_async_context(sse_client(server_url))
            self.stdio, self.write = sse_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()
            print(f"成功连接到SSE服务器: {server_url}")
        except Exception as e:
            print(f"连接到SSE服务器失败: {e}")
            raise

    def get_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        """从LLM获取响应

        Args:
            messages: 消息历史
            tools: 可用的工具列表

        Returns:
            模型响应
        """
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=tools,
            )
            return response
        except Exception as e:
            print(f"获取LLM响应失败: {e}")
            raise

    async def get_tools(self):
        """获取可用工具列表

        Returns:
            工具列表，格式化为OpenAI工具格式
        """
        try:
            response = await self.session.list_tools()
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools]

            return available_tools
        except Exception as e:
            print(f"获取工具列表失败: {e}")
            return []

    async def process_query(self, query: str) -> str:
        """处理查询，使用LLM和可用工具

        Args:
            query: 用户查询

        Returns:
            处理后的响应
        """
        try:
            # 获取相关文档
            relevant_docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # 构建请求
            req = {
                "query": query,
                "context": context
            }

            # 使用对话历史构建消息
            if not self.conversation_history:
                messages = [
                    {
                        "role": "user",
                        "content": json.dumps(req)
                    }
                ]
            else:
                # 加入之前的对话历史
                messages = self.conversation_history.copy()
                # 添加当前查询
                messages.append({
                    "role": "user",
                    "content": json.dumps(req)
                })

            # 获取可用工具
            available_tools = await self.get_tools()

            # 获取LLM响应
            response = self.get_response(messages, available_tools)

            # 处理LLM响应和工具调用
            final_text = []

            for choice in response.choices:
                message = choice.message
                tool_calls = message.tool_calls

                # 如果没有工具调用
                if not tool_calls:
                    final_text.append(message.content)
                    # 更新对话历史
                    self.conversation_history.append({
                        "role": "user",
                        "content": json.dumps(req, ensure_ascii=False)
                    })
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": message.content
                    })
                    continue

                # 处理工具调用
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # print(f"调用工具: {tool_name}, 参数: {json.dumps(tool_args, ensure_ascii=False)}")

                    # 执行工具调用
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"【工具调用】{tool_name}: {result.content}")

                    # 更新对话历史
                    self.conversation_history.append({
                        "role": "user",
                        "content": json.dumps(req, ensure_ascii=False)
                    })

                    # 添加助手的响应和工具调用到历史记录
                    if message.content:
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": message.content
                        })

                    # 将工具调用结果添加到历史记录
                    self.conversation_history.append({
                        "role": "user",
                        "content": result.content
                    })

                    # 获取LLM的进一步回应
                    follow_up_response = self.get_response(self.conversation_history, available_tools)
                    follow_up_content = follow_up_response.choices[0].message.content
                    final_text.append(follow_up_content)

                    # 更新对话历史
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": follow_up_content
                    })

            return "\n".join(final_text)

        except Exception as e:
            print(f"处理查询时出错: {e}")
            raise

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n=== MCP 客户端已启动 ===")
        print("输入你的问题，或输入'quit'退出。")

        while True:
            try:
                query = input("\n问题: ").strip()

                if query.lower() in ['quit', 'exit', '退出']:
                    print("正在退出...")
                    break

                if query.lower() == 'history':
                    print("\n历史查询:")
                    for i, q in enumerate(self.conversation_history, 1):
                        print(f"{i}. {q}")
                    continue

                print("正在处理...")
                response = await self.process_query(query)
                print("\n回答:")
                print("-" * 50)
                print(response)
                print("-" * 50)

            except Exception as e:
                print(f"聊天循环中出错: {e}")

    async def cleanup(self):
        """清理资源"""
        try:
            await self.exit_stack.aclose()
            print("资源已清理")
        except Exception as e:
            print(f"清理资源时出错: {e}")


async def main():
    client = MCPClient()
    try:
        await client.connect_to_sse_server()
        await client.chat_loop()
    except Exception as e:
        print(f"运行主程序时出错: {e}")
        print(f"错误: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

from config import LLM_MODEL, LLM_TEMPERATURE
from retriever import HybridRetriever
from utils import handle_exceptions


@handle_exceptions
def create_qa_chain():
    """创建问答对话链"""
    # 初始化大模型
    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # 初始化记忆
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=1,
        return_messages=True,
        output_key='answer'
    )

    # 构建对话链
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=HybridRetriever(),
        memory=memory,
        chain_type="stuff",
        return_source_documents=False,
    )


def chat_demo():
    """运行聊天演示"""
    qa_chain = create_qa_chain()
    if not qa_chain:
        print("初始化聊天系统失败")
        return

    print("\n===== 本地知识库问答系统 =====")
    print("聊天机器人已启动，输入'exit'退出")
    print("-" * 50)

    while True:
        try:
            query = input("\n用户: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("感谢使用，再见！")
                break

            # 使用invoke方法替代__call__
            result = qa_chain.invoke({"question": query})
            print(f"\nAI: {result['answer']}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")


class CustomChain:
    """自定义链，用于处理特定的问答逻辑"""

    def __init__(self):
        template = """基于以下信息回答问题:
                历史对话: {history}
                上下文: {context}
                问题: {question}
                请用中文详细回答。
                """
        self.memory = ConversationBufferWindowMemory(memory_key="history", k=0)
        retriever = HybridRetriever()
        prompt = ChatPromptTemplate.from_template(template)
        model = OllamaLLM(model=LLM_MODEL)
        self.chain = (
                RunnablePassthrough.assign(
                    history=lambda x: self.memory.load_memory_variables({"question": x["question"]})["history"],
                    context=lambda x: "\n\n".join(
                        [doc.page_content for doc in retriever.get_relevant_documents(x["question"])])
                )
                | prompt
                | model
        )

    def invoke(self, inputs):
        answer = self.chain.invoke({"question": inputs})
        self.memory.save_context({"question": inputs}, {"answer": answer})
        return answer


def test():
    chain = CustomChain()
    print(chain.invoke("云舰的CreateRelease方法的作用是什么?"))
    print("---" * 50)
    print(chain.invoke("都有哪些参数?"))


if __name__ == "__main__":
    chat_demo()

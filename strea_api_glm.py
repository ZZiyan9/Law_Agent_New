# Law_Agent

# 导入必要的库
import streamlit as st # 创建网页应用界面
import os  # 读取环境变量
from langchain_core.output_parsers import StrOutputParser # 处理模型输出结果的解析器
from langchain.prompts import PromptTemplate # 创建文本模板
from langchain.chains import RetrievalQA # 创建检索型问答链
import sys
from zhipuai_embedding import ZhipuAIEmbeddings # ZhipuAI的嵌入模型，将文本转换为向量
from langchain.vectorstores.chroma import Chroma # 向量数据库，存储和检索文本数据
from langchain.memory import ConversationBufferMemory # 绘画记忆管理，跟踪用户与助手的对话历史
from langchain.chains import ConversationalRetrievalChain # 支持带记忆的检索型问答链
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
from zhipuai_llm import ZhipuAILLM

# 从环境变量读取 API 密钥，可以避免泄露
api_key = os.getenv('ZHIPUAI_API_KEY')  # 使用 os.getenv() 获取环境变量

# 生成回答的函数
def generate_response(input_text, api_key):
    llm = ZhipuAILLM(model="glm-4", temperature=0.7, api_key=api_key) # 这里使用glm-4模型
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()  # 解析模型的输出，返回最终的文本答案
    output = output_parser.invoke(output)
    return output

# 向量数据库的创建
def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings() # 定义嵌入模型

    # 读取文本文件（消费者权益保护法.txt）
    with open("消费者权益保护法.txt", "r", encoding="utf-8") as f:
        knowledge_base = f.readlines()  # 读取所有行，形成一个列表，每行是一个段落或文本块
    
    # 清理数据：去掉空行、换行符等
    knowledge_base = [line.strip() for line in knowledge_base if line.strip()]  

    # 将向量数据库保存到本地，能够保证之后再次加载使用时，方便快速，而不是重新计算向量
    # 本地路径
    persist_directory = '../data_base/vector_db/chroma'

    # 使用Chroma 创建并加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 指定存储数据库的位置
        embedding_function=embedding # 将文本转换为向量，存储到数据库中的文本需要全部转换为向量
    )

    # 将清洗后的文本数据添加到向量数据库
    vectordb.add_texts(knowledge_base)  

    return vectordb

# 初始化对话记忆管理
def init_memory():  # 初始化对话的记忆管理系统
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 聊天记录的返回：消息列表，方便追溯每一轮的用户输入和模型的回答
    )
    return memory  


# 带有历史记录的问答链，结合历史、大模型和数据库检索生成回答
def get_chat_qa_chain(question: str, api_key: str, memory: ConversationBufferMemory):
    vectordb = get_vectordb() # 获取向量数据库
    llm = ZhipuAILLM(model="glm-4", temperature=0.1, api_key=api_key) # 封装的ZhipuAILLM

    retriever = vectordb.as_retriever() # 基于向量相似，检索数据库中的相关文本
    qa = ConversationalRetrievalChain.from_llm(  # 问答链
        llm, # 提供的语言模型，生成回答
        retriever = retriever, # 检索
        memory = memory # 记忆
    )
    result = qa({"question": question})  # 将用户问题传给问答链
    return result['answer']


# 不带历史记录的问答链，使用文档和大模型
def get_qa_chain(question: str, api_key: str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model="glm-4", temperature=0.1, api_key=api_key)
    # 构建模板，需要从文档中找答案，没有找到答案，使用自身的知识回答，并告诉用户
    template = "从以下文档中获取信息，回答问题：\n\"\"\"\n{context}\n\"\"\"\n问题是：\n\"\"\"\n{question}\n\"\"\"\n请尽量只使用文档中的直接信息来回答，避免推测。"
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                 template=template)

    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

# Streamlit 应用程序界面
def main():
    st.title('法律援助')

    # 检查是否获取到 API 密钥
    if not api_key:
        st.error("API Key is missing. Please set the ZHIPUAI_API_KEY in your environment variables.")
        return

    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"]
    )

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.memory = init_memory()

    # 初始化记忆
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
            return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
        )

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt, api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt, api_key, st.session_state.memory)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])

if __name__ == "__main__":
    main()

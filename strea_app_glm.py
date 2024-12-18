import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
from zhipuai_llm import ZhipuAILLM

# 从环境变量读取 API 密钥
api_key = os.getenv('ZHIPUAI_API_KEY')  # 使用 os.getenv() 获取环境变量


def generate_response(input_text, api_key):
    llm = ZhipuAILLM(model="glm-4", temperature=0.7, api_key=api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    return output


def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()

    # 读取文本文件（消费者权益保护.txt）
    with open("消费者权益保护.txt", "r", encoding="utf-8") as f:
        knowledge_base = f.readlines()  # 读取所有行，形成一个列表，每行是一个段落或文本块
    
    # 可以进一步处理文本数据，例如：去掉空行，或按需要拆分成句子或段落
    knowledge_base = [line.strip() for line in knowledge_base if line.strip()]  # 去除空行和空白字符

    # 向量数据库持久化路径
    persist_directory = '../data_base/vector_db/chroma'
    # 创建并加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )

    # 将文本数据嵌入并存储到数据库
    vectordb.add_texts(knowledge_base)  # 将文本列表添加到向量数据库中

    return vectordb


def init_memory():
    # 初始化记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    return memory


# 带有历史记录的问答链
def get_chat_qa_chain(question: str, api_key: str, memory: ConversationBufferMemory):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model="glm-4", temperature=0.1, api_key=api_key)

    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']


# 不带历史记录的问答链
def get_qa_chain(question: str, api_key: str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model="glm-4", temperature=0.1, api_key=api_key)
    template = "从文档\n\"\"\"\n{context}\n\"\"\"\n中找问题\n\"\"\"\n{question}\n\"\"\"\n的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n不要复述问题，直接开始回答。"
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
    st.title('🦜🔗 法律援助')

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

    # 侧边栏显示对话记录
    st.sidebar.title("对话记录")
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.sidebar.markdown(f"**用户:** {message['text']}")
            elif message["role"] == "assistant":
                st.sidebar.markdown(f"**助手:** {message['text']}")

    # 主界面消息输入框
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

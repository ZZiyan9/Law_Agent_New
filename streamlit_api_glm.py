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
from datetime import datetime
# 从环境变量读取 API 密钥，可以避免泄露
api_key = os.getenv('ZHIPUAI_API_KEY')  # 使用 os.getenv() 获取环境变量

# 生成回答的函数
def generate_response(input_text, api_key):
    llm = ZhipuAILLM(model="glm-4", temperature=0.7, api_key=api_key) # 这里使用glm-4模型
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()  # 解析模型的输出，返回最终的文本答案
    output = output_parser.invoke(output)
    return output

# 处理文本数据
def get_vectordb():
    # 定义 Embedding
    embedding = ZhipuAIEmbeddings()

    # 本地路径
    persist_directory = '../data_base/vector_db/chroma'

    # 尝试加载已有的数据库，如果没有则创建
    if os.path.exists(persist_directory):
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
    else:
        # 读取文本文件
        with open("消费者权益保护法.txt", "r", encoding="utf-8") as f:
            knowledge_base = f.readlines()  # 读取所有行，形成一个列表，每行是一个段落或文本块
        
        # 清理数据
        knowledge_base = [line.strip() for line in knowledge_base if line.strip()]  

        # 创建新的向量数据库并添加数据
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
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
    # 创建问答链
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
    # 在上一段代码中，并不需要构建这样的模板，因为ConversationBufferMemory已经提供了模板
    template = "从以下文档中获取信息，回答问题：\n\"\"\"\n{context}\n\"\"\"\n问题是：\n\"\"\"\n{question}\n\"\"\"\n请尽量只使用文档中的直接信息来回答，避免推测。"
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                 template=template)
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(), # 检索
                                           return_source_documents=True,  # 返回是哪个文档
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}) # 根据模板格式化输出
    result = qa_chain({"query": question})
    return result["result"]


# streamlit
def main():
    st.title('法护通 - 法律援助')

    # 检查是否获取到 API 密钥
    if not api_key:
        st.error("Error - Please input correct API.")
        return

    # --- 初始化会话存储 ---
    today = datetime.now().strftime("%Y-%m-%d")  # 获取当前日期
    if 'daily_sessions' not in st.session_state:
        st.session_state.daily_sessions = {}  # 每天存储所有会话
    if today not in st.session_state.daily_sessions:
        st.session_state.daily_sessions[today] = []  # 初始化当天的记录

    # 初始化摘要存储
    if 'session_titles' not in st.session_state:
        st.session_state.session_titles = {}  # 存储每个对话的标题

    # --- 新建对话按钮 ---
    st.sidebar.title("📋 对话记录")
    if st.sidebar.button("新建对话"):
        # 创建一个新对话
        new_title = f"新对话 {len(st.session_state.daily_sessions[today]) + 1}"
        st.session_state.daily_sessions[today].append([])  # 添加新的空对话
        if today not in st.session_state.session_titles:
            st.session_state.session_titles[today] = []
        st.session_state.session_titles[today].append(new_title)

    # --- 侧边栏：显示对话标题 ---
    titles = st.session_state.session_titles.get(today, [])
    if titles:  # 显示当天的所有对话标题
        selected_title = st.sidebar.radio("选择对话", titles)
        selected_index = titles.index(selected_title)
    else:
        selected_title, selected_index = None, -1

    # --- 模式选择 ---
    selected_method = st.radio(
        "您要选择什么样的检索问答方式",
        ["None", "qa_chain", "chat_qa_chain"],
        captions=["不使用RAG的普通模式", "不带历史记录的RAG", "带历史记录的RAG"]
    )

    # 初始化当前对话历史和记忆
    if 'memory' not in st.session_state:
        st.session_state.memory = init_memory()

    # 当前对话记录
    current_messages = []
    if selected_index != -1:  # 如果选择了历史对话
        current_messages = st.session_state.daily_sessions[today][selected_index]

    # --- 聊天窗口 ---
    st.subheader(f"当前对话：{selected_title or '新对话'}")
    messages = st.container(height=300)  # 聊天容器

    # 聊天输入框
    if prompt := st.chat_input("请输入您的问题"):
        # 记录用户输入
        if selected_index == -1:  # 新对话
            current_messages = [{"role": "user", "text": prompt}]  # 初始化当前对话记录
            st.session_state.daily_sessions[today].append(current_messages)

            # 设置对话标题为第一条用户消息
            title = prompt  # 使用用户的第一条消息作为标题
            if today not in st.session_state.session_titles:
                st.session_state.session_titles[today] = []
            st.session_state.session_titles[today].append(title)
        else:  # 更新已有对话
            current_messages.append({"role": "user", "text": prompt})

        # 生成回答
        if selected_method == "None":
            answer = generate_response(prompt, api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt, api_key)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt, api_key, st.session_state.memory)

        # 记录助手回复
        if answer:
            current_messages.append({"role": "assistant", "text": answer})

        # 更新当天的记录
        if selected_index == -1:  # 新对话
            st.session_state.daily_sessions[today][-1] = current_messages
        else:  # 更新已有对话
            st.session_state.daily_sessions[today][selected_index] = current_messages

        # 显示当前对话
        for msg in current_messages:
            if msg["role"] == "user":
                messages.chat_message("user").write(msg["text"])
            elif msg["role"] == "assistant":
                messages.chat_message("assistant").write(msg["text"])

if __name__ == "__main__":
    main()

import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, \
    ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
import base64
import json
from datetime import datetime
import random

# 现在我直接找了一个文件问答的模板，后期只需要把我们的pipeline 封装成函数，即可执行用户输入对应输出
# 目前可以做的UI设计：
# 继续完善课件回答chatbot
# 设计复习chatbot的产品展示形式  √
# 将两者放在一个网站 (侧边栏 or 其他方法)  √
# 网站配色 可参考这个网站：https://coolors.co/262322-63372c-c97d60-ffbcb5-f2e5d7
# 修改streamlit配色方法：https://blog.csdn.net/BigDataPlayer/article/details/128962594
# 本地调试，命令行进入相应路径执行： streamlit run app.py

# 更新：目前复习模块的展示已经完成，但还剩相关问题生成需对接，包括聊天记录存储也已经完成，等待接口完成最终调试


# Load environment variables
load_dotenv()

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"
CHAT_HISTORY_DIR = "chat_history"  # New directory for chat history

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)  # Ensure chat history directory exists


def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)


def handle_query(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
        (
            "user",
            """You are a Q&A assistant named CHATTO, created by Suriya. You have a specific response programmed for when users specifically ask about your creator, Suriya. The response is: "I was created by Suriya, an enthusiast in Artificial Intelligence. He is dedicated to solving complex problems and delivering innovative solutions. With a strong focus on machine learning, deep learning, Python, generative AI, NLP, and computer vision, Suriya is passionate about pushing the boundaries of AI to explore new possibilities." For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
            Context:
            {context_str}
            Question:
            {query_str}
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)

    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."
    
    
# 随机选择一个QA文件，然后从该文件中随机选择一个QA对话
def get_random_qa_pair():
    chat_files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")]

    if not chat_files:
        return None, None  # 如果没有文件，返回None

    # 从目录中随机选择一个文件
    selected_file = random.choice(chat_files)

    # 读取文件并提取所有聊天记录
    with open(f"{CHAT_HISTORY_DIR}/{selected_file}", 'r') as file:
        chat_data = json.load(file)

    # 确保文件中有内容
    if len(chat_data) > 0:
        # 从聊天记录中随机选择一个QA对
        random_index = random.randint(0, len(chat_data) - 1)
        selected_qa = chat_data[random_index]
        return selected_qa['user'], selected_qa['answer']
    
    return None, None  # 如果文件中没有聊天记录


# Streamlit app initialization
st.title("Chat with your PDF📄")
# st.markdown("Built by [Qichen❤️]()")
st.markdown("chat here👇")

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', "content": 'Hello! Upload a PDF and ask me anything about its content.'}]

with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            filepath = "data/saved_pdf.pdf"
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # displayPDF(filepath)  # Display the uploaded PDF
            data_ingestion()  # Process PDF every time new file is uploaded
            st.success("Done")
    
# New code from here
    if st.button("Review Mode"):
        # 获取随机QA对
        user, answer = get_random_qa_pair()
        print(user,answer)
        
        if user is not None and answer is not None:
            # 存储到会话状态中
            st.session_state['current_qa_pair'] = {'user': user, 'answer': answer}
            st.rerun()  # 触发刷新以更新显示
        else:
            st.write("No QA pairs found.")

# If Review Mode is activated and we have a QA pair
if 'current_qa_pair' in st.session_state:
    # Refresh button to get a new random QA pair
    if st.sidebar.button("Refresh QA"):
        # 获取新的随机QA对
        user, answer = get_random_qa_pair()
        
        if user is not None and answer is not None:
            st.session_state['current_qa_pair'] = {'user': user, 'answer': answer}
            st.rerun()  # 触发刷新
        else:
            st.sidebar.write("No QA pairs found.")

    st.sidebar.write("Random QA Pair:")
    qa_pair = st.session_state['current_qa_pair']
    user = qa_pair.get("user", "Unknown")
    answer = qa_pair.get("answer", "No answer")
    st.sidebar.write(f"User: {user}")
    st.sidebar.write(f"Answer: {answer}")

# New code ends here

# Save chat history
def auto_save_conversation(messages):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history/conversation_{timestamp}.json"
    with open(filename, 'w') as file:
        json.dump(messages, file, indent=4)
    # return filename

if 'all_QA' not in st.session_state:
    st.session_state.all_QA = []

if 'qa_id' not in st.session_state:
    st.session_state.qa_id = 0

user_prompt = st.chat_input("Ask me anything about the content of the PDF:")
if user_prompt:
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = handle_query(user_prompt)
    st.session_state.messages.append({'role': 'assistant', "content": response})

    st.session_state.all_QA.append({
        "id": st.session_state.qa_id,
        "user": user_prompt,
        "answer": response 
    })
    st.session_state.qa_id += 1

    filename = auto_save_conversation(st.session_state.all_QA)
    st.success(f"Conversation auto-saved to {filename}")

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
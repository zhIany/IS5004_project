import gradio as gr
import os

# 假设我们有一个简单的聊天机器人函数
def chat_bot(input_text, history, uploaded_file):
    # 这里我们只是简单返回上传文件的名称和用户的输入
    if uploaded_file:
        response = f"Answer related to {os.path.basename(uploaded_file.name)}"
    else:
        response = "I don't have enough information to answer that."
    # 更新历史记录
    history.append((input_text, response))
    return response, history

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## Data Science Chat Bot")
    gr.Markdown("Upload a document and ask questions related to it.")

    # 文件上传组件
    file_input = gr.File(label="Upload Document")

    # 聊天历史记录
    chat_history = gr.State([])

    # 聊天输入和输出
    with gr.Row():
        chat_input = gr.Textbox(label="Ask a Question", placeholder="Type your question here...")
        chat_output = gr.Textbox(label="Answer", placeholder="Waiting for your question...")

    # 提交按钮
    submit_button = gr.Button("Submit")

    # 查询历史记录展示
    history_output = gr.Textbox(label="Query History", lines=10)

    # 设置按钮点击事件
    submit_button.click(
        fn=chat_bot,
        inputs=[chat_input, chat_history, file_input],
        outputs=[chat_output, chat_history]
    )

    # 更新查询历史记录展示
    def update_history(history):
        # 将历史记录转换为字符串格式
        history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
        return history_str

    # 在聊天历史记录更新时触发
    chat_history.change(
        fn=update_history,
        inputs=chat_history,
        outputs=history_output
    )

# 运行 Gradio 应用
demo.launch()

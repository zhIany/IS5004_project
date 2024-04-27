import gradio as gr
import os

# 假设我们有一个简单的聊天机器人函数
def chat_bot(input_text, history, uploaded_file):
    # 这里我们只是简单返回上传文件的名称和用户的输入
    if uploaded_file:
        response = f"Answer related to {os.path.basename(uploaded_file.name)}"
    else:
        response = "I don't have enough information to answer that."
    return response, history + [(input_text, response)]

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
    history_output = gr.JSON(label="Query History")

    # 设置按钮点击事件
    submit_button.click(
        fn=chat_bot,
        inputs=[chat_input, chat_history, file_input],
        outputs=[chat_output, chat_history]
    )

    # 更新查询历史记录
    def update_history(history):
        return history

    chat_history.change(
        fn=update_history,
        inputs=chat_history,
        outputs=history_output
    )

# 运行 Gradio 应用
demo.launch()

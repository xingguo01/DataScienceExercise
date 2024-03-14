import gradio as gr
import random
import time
#Only works with a valid OpenAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
#from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# load word embeddings from database
persist_directory = "db"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)


# memory for chat history ... needed for gradio
memory = ConversationBufferMemory (
    memory_key="chat_history",
    return_messages=False,
)

# the bot
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0, max_tokens=-1), 
    chain_type="stuff",
    retriever=db.as_retriever(),
    memory=memory,
    get_chat_history=lambda h: h, 
    verbose=True,
)

# UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = qa.run({"question": history[-1][0], "chat_history": history[:-1]})
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

#demo.launch(share=True)
demo.launch()


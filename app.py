import os
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import logging
import sys
from llama_index.core import StorageContext, load_index_from_storage

# logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
documents = SimpleDirectoryReader("./test_document").load_data()

# indexing
index = VectorStoreIndex.from_documents(documents, show_progress = True)

# save index
index.storage_context.persist(persist_dir="./index")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./index")

# load index
index = load_index_from_storage(storage_context)

retriever = index.as_retriever(
similarity_top_k = 10
)

question = """ You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:\
1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.

Question: 请为我解释课件第10页seq to seq结构图的含义
"""

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

response = query_engine.query(question)

# test retriever result
nodes = retriever.retrieve(question)
for i in range(10):
    print("label page ", nodes[i].node.metadata["page_label"])

# test result
page = 0
sum_correct = 0
for i in range(1,68):
    page_list = []
    question =  f"""
    You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:\
    1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
    2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.
    Question: 请为我总结课件第{i}页的主要内容
    """
    nodes = retriever.retrieve(question)
    for j in range(10):
        page_list.append(int(nodes[j].node.metadata["page_label"]))
#         print("label page ", nodes[j].node.metadata["page_label"])
    print(f"label list for page {i}", page_list)
    if i in page_list:
        sum_correct = sum_correct + 1
        print(f"page {i} 索引正确")
print("正确率", (sum_correct/67)*100, "%")
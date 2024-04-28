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

## 最普通的RAG 效果比较差舍弃
# documents = SimpleDirectoryReader("./dataset").load_data()
# # indexing
# index = VectorStoreIndex.from_documents(documents, show_progress = True)
#
# # save index
# index.storage_context.persist(persist_dir="./index")
#
# # rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="./index")
#
# # load index
# index = load_index_from_storage(storage_context)
#
# retriever = index.as_retriever(
# similarity_top_k = 10
# )
#
# question = """ You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:\
# 1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
# 2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.
#
# Question: 请为我解释课件第10页seq to seq结构图的含义
# """
#
# # configure response synthesizer
# response_synthesizer = get_response_synthesizer()
#
# # assemble query engine
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
#     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
# )
#
# response = query_engine.query(question)
#
# # test retriever result
# nodes = retriever.retrieve(question)
# for i in range(10):
#     print("label page ", nodes[i].node.metadata["page_label"])
#
# # test result
# page = 0
# sum_correct = 0
# for i in range(1, 49):
#     page_list = []
#     question =  f"""
#     You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:\
#     1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
#     2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.
#     Question: 请为我总结课件第{i}页的主要内容
#     """
#     nodes = retriever.retrieve(question)
#     for j in range(10):
#         page_list.append(int(nodes[j].node.metadata["page_label"]))
# #         print("label page ", nodes[j].node.metadata["page_label"])
#     print(f"label list for page {i}", page_list)
#     if i in page_list:
#         sum_correct = sum_correct + 1
#         print(f"page {i} 索引正确")
# print("正确率", (sum_correct/67)*100, "%")



# # knowledge graph index 知识图谱RAG，将PDF课件以知识图谱的形式作为索引，使用RAG来回答问题
# from llama_index.core import VectorStoreIndex, get_response_synthesizer
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
# from llama_index.core import StorageContext
#
# graph_store = SimpleGraphStore()
# storage_context = StorageContext.from_defaults(graph_store=graph_store)
#
# # NOTE: can take a while!
# index = KnowledgeGraphIndex.from_documents(
#     documents,
#     max_triplets_per_chunk=2,
#     storage_context=storage_context,
# )
# graph_rag_retriever = index.as_retriever()
#
# question =  """
#     You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:\
# 1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
# 2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.
#
#
# Question: 请为我解释课件第10页seq to seq结构图的含义
# """
# # configure response synthesizer
# response_synthesizer = get_response_synthesizer()
#
# # assemble query engine
# query_engine = RetrieverQueryEngine(
#     retriever=graph_rag_retriever,
#     response_synthesizer=response_synthesizer,
#     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
# )
# response = query_engine.query(question)
#
#
# try PDF parser
# from pathlib import Path
#
# from llama_index.readers.file import PyMuPDFReader
#
# loader = PyMuPDFReader()
# documents = loader.load_data(file_path=Path("./test_document/TPML_day_2_v3.pdf"), metadata=True)
# indexing
# index_pdf = VectorStoreIndex.from_documents(documents, show_progress = True)
#





# RAG + rerank 方法，
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./db").load_data()
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# splitter = SentenceSplitter(chunk_size=256)

index = VectorStoreIndex.from_documents(documents)

print(index)


from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=2)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=10)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=10
)

from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=12,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=True,
    query_gen_prompt=
        "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n",  # we could override the query generation prompt here
)


from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)
# apply nested async to run in a notebook
import nest_asyncio
nest_asyncio.apply()

question = """ You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:
    1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
    2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.


    Question: 请为我解释课件第{i}页的含义"""

nodes = retriever.retrieve(question)
print(nodes[0])

#
# for i in range(1, 69):
#     question = f"""
#         You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:\
#     1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
#     2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.
#
#
#     Question: 请为我解释课件第{i}页的含义
#     """
#     response = query_engine.query(question)
#     nodes = retriever.retrieve(question)
#     for j in range(10):
#         print("label page ", nodes[j].node.metadata["page_label"])
#     print(response)

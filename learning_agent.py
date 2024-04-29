import logging
import sys
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
# logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# RAG + rerank 方法，
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import nest_asyncio
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core import StorageContext
# try PDF parser
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader


# 普通导入数据存储在index中
def ingest_and_index(data_dir, PERSIST_DIR):
    documents = SimpleDirectoryReader(data_dir).load_data()
    splitter = SentenceSplitter(chunk_size=256)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter], show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# 使用PDF READER 导入数据
def ingest_and_index_with_pdf_reader(data_path):
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path=Path(data_path), metadata=True)
    index_pdf = VectorStoreIndex.from_documents(documents, show_progress = True)
    return index_pdf

# 使用 knowledge graph 存储导入后的index，好像还有其他load_index方法可以尝试
def index_and_ingest_knowledge_graph(data_dir):
    documents = SimpleDirectoryReader(data_dir.load_data())
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # NOTE: can take a while!
    index_graph = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=2,
        storage_context=storage_context,
    )
    print("graph index done")
    return index_graph


# 重排 index 的retrieve 方法
def retriever_rerank(index):
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=10
    )
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
    print(retriever)
    print("done")
    return retriever

# knowledge graph retriever
def retriever_knowledge_graph(index_g):
    graph_rag_retriever = index_g.as_retriever()
    print("graph retrieve done")
    return graph_rag_retriever

# 普通 query engine
def query_normal(retriever):
    query_engine = RetrieverQueryEngine.from_args(retriever)
    # apply nested async to run in a notebook
    nest_asyncio.apply()

    question = """ You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:
        1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
        2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.
  
        Question: 请为我解释课件第{i}页的含义"""
    nodes = retriever.retrieve(question)
    response = query_engine.query(question)
    return response, nodes


def query_knowledge_graph(graph_rag_retriever):
    question =  """
        You are a skilled teaching assistant with 10 years of experience in Artificial Intelligence, particularly Natural Language Processing. You have two main job responsibilities:\
    1. Discover the parts of the course lesson plan that students are more interested in through dialogue with them. At the same time, you will combine your knowledge to introduce the relevant content to the students, so that the students can have a deeper understanding of the relevant content in the lesson plan that they are interested in or don't know much about yet.
    2. Answer students' questions about AI and practical language processing according to the course handout, trying to explain it step by step to ensure that students can fully understand all the concepts or related applications, and give some code as examples if necessary. If you can find relevant information in the course handouts, please try to answer based on your own knowledge and experience. Remember not to give incorrect answers. If you are unsure of an answer, tell the student that you are unsure. If the answer is correct and accurate, you will be paid well. Therefore, please try to address all questions asked by the student.
    
    
    Question: 请为我解释课件第10页seq to seq结构图的含义
    """
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=graph_rag_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )
    response = query_engine.query(question)
    nodes = graph_rag_retriever.retrieve(question)
    return response, nodes

# def save_history_data(queries, response, keyword):


if __name__ == "__main__":
    data_dir = "dataset"
    per_dir = "db"
    print("start indexing ...")
    index = ingest_and_index(data_dir, per_dir)
    print("start retrieve")
    retriever = retriever_rerank(index)

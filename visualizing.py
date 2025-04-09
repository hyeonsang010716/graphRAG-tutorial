"""
Load dataset
"""
from graph_rag_example_helpers.datasets.animals import fetch_documents
animals = fetch_documents()


"""
Load ChromaDB
"""
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_graph_retriever.transformers import ShreddingTransformer
from langchain_community.vectorstores.utils import filter_complex_metadata

from dotenv import load_dotenv
load_dotenv()

shredder = ShreddingTransformer() 
shredded_docs = list(shredder.transform_documents(animals))
filtered_docs = filter_complex_metadata(shredded_docs) # Filter Dict Metada
vector_store = Chroma.from_documents(
    documents=filtered_docs,
    embedding=OpenAIEmbeddings(),
    collection_name="animals",
)


"""
Load Graph Retriever
"""
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.adapters.chroma import ChromaAdapter

simple = GraphRetriever(
    store = ChromaAdapter(vector_store, shredder, {"keywords"}),
    edges = [("habitat", "habitat"), ("origin", "origin"), ("keywords", "keywords")],
    strategy = Eager(k=10, start_k=1, max_depth=3),
)


"""
Run Graph Retriever
"""
simple_results = simple.invoke("what mammals could be found near a capybara")

for doc in simple_results:
    print(f"{doc.id}: {doc.page_content}")
    

"""
Visualizing GraphDB
"""
import networkx as nx
import matplotlib.pyplot as plt
from langchain_graph_retriever.document_graph import create_graph

document_graph = create_graph(
    documents=simple_results,
    edges = simple.edges,
)

nx.draw(document_graph, with_labels=True)
plt.show()
import numpy as np
import faiss
from annoy import AnnoyIndex

# Function to perform faiss search
def faiss_search(embeddings, query_vector, k):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(np.array([query_vector]), k)
    return distances[0], indices[0]

# Function to annoy search
def annoy_search(embeddings, query_vector, k):
    dimension = embeddings.shape[1]
    index = AnnoyIndex(dimension, 'angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10) # 10 trees - more trees gives higher precision
    indices, distances = index.get_nns_by_vector(query_vector, k, include_distances=True)
    return distances, indices



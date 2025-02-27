from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# Word Embeddings
def word_embeddings():
    # Use Static Word Embedding called Word2Vec
    sentences = [['cat', 'say', 'meow'],['dog', 'say', 'woof']]
    model = Word2Vec(sentences=sentences, vector_size=10, window=5, min_count=1, workers=4)
    # Parameters:
    # - vector_size=10: Dimensionality of the word vectors
    # - window=5: Maximum distance between current and predicted word within a sentence
    # - min_count=1: Ignores all words with total frequency lower than this
    # - workers=4: Number of CPU cores to use for training
    print(f"Word Embedding for Cat is - {model.wv['cat']}")

# Sentence Embeddings
def sentence_embeddings():
    # SentenceTransformer is a library for state-of-the-art sentence embeddings
    # It's based on BERT architecture and fine-tuned for generating sentence embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2') # paraphrase-MiniLM-L6-v2 is the pretrained model being used
    sentences = ["This is an example Sentence", "Each Sentence is converted to a Vector"]
    embeddings = model.encode(sentences)
    print(f"Sentence Embedding Shape is - {embeddings.shape}")
    print(f"First Sentence Embedding is - {embeddings[0][:5]}") # We display only first 5 dimensions

if __name__ == '__main__':
    print("Word Embeddings")
    word_embeddings()
    print("Sentence Embeddings")
    sentence_embeddings()

from flair.embeddings import FlairEmbeddings,WordEmbeddings,PooledFlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence

embeddings=StackedEmbeddings([
        WordEmbeddings('glove'),

        FlairEmbeddings('mix-forward'),
        FlairEmbeddings('pubmed-forward'),

        PooledFlairEmbeddings('news-forward'),

        ])

sentence='i wish a champion'
sentence=Sentence(sentence)
embeddings.embed(sentence)

embedding_type=[
        WordEmbeddings('glove'),

        FlairEmbeddings('mix-forward'),
        FlairEmbeddings('pubmed-forward'),
        PooledFlairEmbeddings('news-forward'),
        ]
print([i.embedding_length for i in embedding_type])

print(embeddings.embedding_length)
print(sentence.tokens[0].embedding.shape)

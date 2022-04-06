import pandas as pd
from gensim.models import Word2Vec
import os

def create_words(seqs, wordsize=1):
    """
    Create a list of list of with words of size wordsize for each input sequence
    """
    seq_list = []
    for seq in seqs:
        word_list = []
        for i in range(0, len(seq)-wordsize+1):
            word_list.append(seq[i:i+wordsize])
        seq_list.append(word_list)
    return seq_list


def main():
    FEATURE = "cdr3a"
    DIR = os.path.join("/Users/christianjohansen/Desktop/speciale/processing")
    DATA_FILE = os.path.join(DIR, "out/10x_200000sample.csv")
    EMBEDDING_FILE = os.path.join(DIR, "out/embeddings_cdr3a.wv")

    df = pd.read_csv(DATA_FILE)
    cdr = df[FEATURE].to_list()
    cdr_words = create_words(cdr, wordsize=1)
    model = Word2Vec(cdr_words,
                     vector_size=100,
                     window=20,
                     min_count=1,
                     workers=4,
                     epochs=500)
    print()
    print(model.wv.most_similar("A"))
    print()
    print(model.wv.most_similar("W"))
    print()
    print(model.wv.most_similar("D"))
    print()
    print(model.wv.most_similar("R"))
    model.wv.save(EMBEDDING_FILE)



if __name__ == "__main__":
    main()
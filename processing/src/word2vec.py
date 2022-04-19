import pandas as pd
from gensim.models import Word2Vec
import os
from Bio import SeqIO

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

def load_csv_file(filename, feature):
    df = pd.read_csv(filename)
    return df[feature].to_list()

def load_fasta_file(filename, n=20000):
    """
    Get n first sequences from the supplied fastafile
    """
    sequences = []
    for i, record in enumerate(SeqIO.parse(filename, 'fasta')):
        if i == n:
            break
        sequences.append(str(record.seq))
    return sequences




def main():
    FEATURE = "cdr3a"
    DIR = os.path.join("/Users/christianjohansen/Desktop/speciale/processing")
    DATA_FILE = os.path.join(DIR, "out/10x_200000sample.csv")
    FASTA_FILE = os.path.join(DIR, "data/raw/uniprot_sprot.fasta")
    EMBEDDING_FILE = os.path.join(DIR, "out/embeddings_proteins.wv")
    #sequences = load_csv_dat(filename=DATA_FILE, feature=FEATURE)
    sequences = load_fasta_file(FASTA_FILE, 1000000)
    words = create_words(sequences, wordsize=1)
    model = Word2Vec(words,
                     vector_size=20,
                     window=11,
                     min_count=1,
                     workers=4,
                     epochs=20)
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
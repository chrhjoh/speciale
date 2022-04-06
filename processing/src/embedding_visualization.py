from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    DIR = os.path.join("/Users/christianjohansen/Desktop/speciale/processing")
    EMBEDDING_FILE = os.path.join(DIR, "out/embeddings_cdr3b.wv")
    AMINO_ACIDS = ["A",  "R",  "N",  "D", "C",  "Q",  "E",  "G", "H",  "I",  "L",  "K",  "M",  "F",  "P",  "S",  "T", "W",  "Y",  "V"]
    weight_vector = KeyedVectors.load(EMBEDDING_FILE)
    embeddings = [weight_vector[aa] for aa in AMINO_ACIDS]
    embeddings = np.stack(embeddings)
    model = PCA(2)
    emb_2d = model.fit_transform(embeddings)

    fig, ax = plt.subplots()
    ax.scatter(emb_2d[:,0], emb_2d[:,1])
    for i in range(len(AMINO_ACIDS)):
        ax.annotate(AMINO_ACIDS[i],(emb_2d[i,0], emb_2d[i,1]))
    plt.show()

    print("hello")

if __name__ == "__main__":
    main()

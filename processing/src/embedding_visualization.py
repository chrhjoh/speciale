from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    DIR = os.path.join("/Users/christianjohansen/Desktop/speciale/processing")
    EMBEDDING_FILE = os.path.join(DIR, "out/embeddings_proteins.wv")
    AMINO_ACIDS = ["A",  "R",  "N",  "D", "C",  "Q",  "E",  "G", "H",  "I",  "L",  "K",  "M",  "F",  "P",  "S",  "T", "W",  "Y",  "V"]
    FEATURE = ["hydrophobic", "positive", "polar", "negative", "unique", "polar", "negative", "unique", "polar", "hydrophobic", "hydrophobic",
               "positive", "hydrophobic", "aromatic", "unique", "polar", "polar", "aromatic", "aromatic", "hydrophobic"]
    FEAT_TO_COL = {"hydrophobic" : "green",
                   "polar" : "purple",
                   "positive": "blue",
                   "negative": "red",
                   "aromatic": "black",
                   "unique": "yellow"}
    COLORS = [FEAT_TO_COL[feat] for feat in FEATURE]
    weight_vector = KeyedVectors.load(EMBEDDING_FILE)
    embeddings = [weight_vector[aa] for aa in AMINO_ACIDS]
    embeddings = np.stack(embeddings)
    model = TSNE(perplexity=3,
                 learning_rate=5)
    emb_2d = model.fit_transform(embeddings)

    fig, ax = plt.subplots()
    ax.scatter(emb_2d[:,0], emb_2d[:,1], color=COLORS)
    for i in range(len(AMINO_ACIDS)):
        ax.annotate(AMINO_ACIDS[i],(emb_2d[i,0], emb_2d[i,1]))
    plt.show()


if __name__ == "__main__":
    main()

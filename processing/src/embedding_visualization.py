from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    DIR = os.path.join("/Users/christianjohansen/Desktop/speciale/processing")
    EMBEDDING_FILE = os.path.join(DIR, "out/embeddings_cdr3b.wv")
    AMINO_ACIDS = ["A",  "R",  "N",  "D", "C",  "Q",  "E",  "G", "H",  "I",  "L",  "K",  "M",  "F",  "P",  "S",  "T", "W",  "Y",  "V"]
    FEATURE = ["Hydrophobic", "Positive", "Polar", "Negative", "Unique", "Polar", "Negative", "Unique", "Polar", "Hydrophobic", "Hydrophobic",
               "Positive", "Hydrophobic", "Aromatic", "Unique", "Polar", "Polar", "Aromatic", "Aromatic", "Hydrophobic"]
    FEAT_TO_COL = {"Hydrophobic" : "green",
                   "Polar" : "purple",
                   "Positive": "blue",
                   "Negative": "red",
                   "Aromatic": "black",
                   "Unique": "yellow"}
    COLORS = [FEAT_TO_COL[feat] for feat in FEATURE]
    weight_vector = KeyedVectors.load(EMBEDDING_FILE)
    embeddings = [weight_vector[aa] for aa in AMINO_ACIDS]
    embeddings = np.stack(embeddings)

    model = TSNE(perplexity=2,
                 learning_rate=3,
                 random_state=123)
    emb_2d = model.fit_transform(embeddings)
    df = pd.DataFrame({"x" : emb_2d[:,0], "y" : emb_2d[:, 1], "feature": FEATURE})

    fig, ax = plt.subplots()
    for key, group in df.groupby("feature"):
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=FEAT_TO_COL[key])
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    for i in range(len(AMINO_ACIDS)):
        ax.annotate(AMINO_ACIDS[i],(emb_2d[i,0], emb_2d[i,1]))
    plt.show()


if __name__ == "__main__":
    main()

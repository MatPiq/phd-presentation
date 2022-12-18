import os
import pickle
from collections import defaultdict

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from process_images import Img


def get_image(img, zoom=0.05):
    """Helper to offset an image"""
    return OffsetImage(np.asarray(img), zoom=zoom)


def create_graph(imgs):

    G = nx.Graph()
    # Append edges
    for img in imgs:
        for edge in img.labels:
            G.add_edge(img.idx, edge)

    return G


def bipartite_graph(G, imgs):

    # Get layout
    pos = nx.spring_layout(G, seed=42, iterations=100)

    # Get subgraph of degree > 1
    G_sub = G.subgraph([n for n, d in G.degree() if d > 1])

    fig, ax = plt.subplots(dpi=1000, figsize=(5, 5))

    nx.draw_networkx_edges(G_sub, pos, alpha=0.5, width=0.1, ax=ax)
    nx.draw_networkx_labels(G_sub, pos, font_size=3, ax=ax)

    for n in G:
        if isinstance(n, int):
            x0, y0 = pos[n]  # figure coordinates
            ab = AnnotationBbox(get_image(imgs[n].img, 0.025), (x0, y0), frameon=False)
            ax.add_artist(ab)

    ax.set_title("Bipartite Network")
    fig.savefig("figs/bipartite.pdf")


def communities(G, imgs):

    partition = community_louvain.best_partition(G, random_state=42)
    partition = {n: c for n, c in partition.items() if isinstance(n, int)}
    communities = defaultdict(list)
    for img in imgs:
        communities[partition[img.idx]].append(img.img)

    for i, cluster in enumerate(communities.values(), start=1):

        fig, axs = plt.subplots(ncols=len(cluster), dpi=1000)

        for j, ax in enumerate(axs):
            ax.imshow(np.asarray(cluster[j]))
            ax.axis("off")

        fig.set_tight_layout(tight=True)
        fig.set_constrained_layout(constrained=True)
        fig.savefig(f"tmp_figs/community_{i}", bbox_inches="tight")
        plt.close()

    files = [f for f in os.listdir("tmp_figs") if f.endswith(".png")]
    fig, axs = plt.subplots(nrows=len(files), figsize=(10, 10), dpi=500)
    for i, ax in enumerate(axs):
        ax.set_axis_off()
        filename = "tmp_figs/" + files[i]
        ax.imshow(mpimg.imread(filename))
        ax.set_title(f"Community {i+1}")
    fig.tight_layout()
    fig.savefig("figs/communities.pdf")


if __name__ == "__main__":
    with open("processed_images.pkl", "rb") as f:
        imgs = pickle.load(f)
    G = create_graph(imgs)

    bipartite_graph(G, imgs)
    communities(G, imgs)

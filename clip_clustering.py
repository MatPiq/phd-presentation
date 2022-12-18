import contextlib
import os
import pickle
from collections import defaultdict

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from process_images import Img


def get_image(img, zoom=0.05):
    """Helper to offset an image"""
    return OffsetImage(np.asarray(img), zoom=zoom)


def clip_embeddings(imgs):
    """
    Creates a matrix of Embeddings
    """
    embeds = np.empty(shape=(len(imgs), 768))
    for img in imgs:
        embeds[img.idx] = img.embedding

    return embeds


def tsne_projection(embeds):

    # Fit and rescale
    xy = TSNE(n_components=2, random_state=42).fit_transform(embeds) * 0.05

    fig, ax = plt.subplots(figsize=(5, 5), dpi=800)
    ax.scatter(xy[:, 0], xy[:, 1])

    for x0, y0, img in zip(xy[:, 0], xy[:, 1], imgs):
        ab = AnnotationBbox(get_image(img.img), (x0, y0), frameon=False)
        ax.add_artist(ab)

    ax.set_title("tSNE CLIP Embeddings")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    fig.savefig("figs/clip_proj.pdf", bbox_inches="tight")


def kmeans_clusters(embeds, imgs):

    fit = KMeans(n_clusters=9, random_state=42, n_init="auto").fit(embeds)

    clusters = defaultdict(list)
    for i, lab in enumerate(fit.labels_):
        clusters[lab].append(imgs[i].img)

    # fig, axs1 = plt.subplots(nrows = len(communities))
    for f in os.listdir("tmp_figs"):
        if f.startswith("cluster"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(f)

    for i, cluster in enumerate(clusters.values(), start=1):

        fig, axs = plt.subplots(ncols=len(cluster), dpi=1000)

        for j, ax in enumerate(axs):
            ax.imshow(np.asarray(cluster[j]))
            ax.axis("off")

        fig.set_tight_layout(tight=True)
        fig.set_constrained_layout(constrained=True)
        fig.savefig(f"tmp_figs/cluster_{i}", bbox_inches="tight")
        plt.close()

    files = [f for f in os.listdir("tmp_figs") if f.startswith("cluster")]
    fig, axs = plt.subplots(nrows=len(files), figsize=(10, 10), dpi=500)
    for i, ax in enumerate(axs):
        ax.set_axis_off()
        filename = f"tmp_figs/{files[i]}"
        ax.imshow(mpimg.imread(filename))
        ax.set_title(f"Cluster {i+1}")

    fig.tight_layout()
    fig.savefig("figs/clusters.pdf")


if __name__ == "__main__":
    with open("processed_images.pkl", "rb") as f:
        imgs = pickle.load(f)

    embeds = clip_embeddings(imgs)

    tsne_projection(embeds)
    kmeans_clusters(embeds, imgs)

import io
import os
import pickle
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import List

import clip
import imagehash
import numpy as np
import torch
from google.cloud import vision
from PIL import Image
from PIL.PngImagePlugin import PngImageFile


@dataclass
class Img:
    idx: int
    img: PngImageFile
    path: PosixPath
    labels: List[str]
    embedding: np.ndarray
    phash: imagehash.ImageHash


def get_client():
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] = "uplifted-sol-371311-43b56e7ae1bf.json"
    return vision.ImageAnnotatorClient()


def process_images() -> List[Img]:

    imgs = []

    client = get_client()
    model, preprocess = clip.load("ViT-L/14", device="cpu")

    for i, img_path in enumerate(Path("images").glob("*.png")):

        with io.open(img_path, "rb") as image_file:
            content = image_file.read()
            img_objdet = vision.Image(content=content)

            # Extract labels with Google Vision
            response = client.label_detection(image=img_objdet)
            labels = [label.description for label in response.label_annotations]

        img = Image.open(img_path)
        # Extract embeddings with OpenAI Clip
        with torch.no_grad():
            imgp = preprocess(img).unsqueeze(0)
            embedding = model.encode_image(imgp).numpy()

        phash = imagehash.phash(img)

        imgs.append(
            Img(
                idx=i,
                img=img,
                path=img_path,
                labels=labels,
                embedding=embedding,
                phash=phash,
            )
        )

    return imgs


if __name__ == "__main__":
    imgs = process_images()
    path = "processed_images.pkl"
    print(f"Writing {len(imgs)} to {path}")
    with open(path, "wb") as f:
        pickle.dump(imgs, f)

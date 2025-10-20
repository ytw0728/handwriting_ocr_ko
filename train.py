import os
from src.data.stream import StreamingDataLoader
from src.data.load_label import *

def train_step(imgs_batch, metas_batch):
    print("Train!")

if __name__ == "__main__":
    label_files = [os.path.splitext(f)[0] for f in os.listdir("data/train/labels") if f.endswith(".json")]

    train_loader = StreamingDataLoader(label_files, batch_size=4, prefetch_batches=2, use_meta=True)
    try:
        for imgs_batch, metas_batch in train_loader:
            print("Train batch:", imgs_batch.shape, "Meta count:", len(metas_batch))
            train_step(imgs_batch, metas_batch)
    finally:
        train_loader.shutdown()
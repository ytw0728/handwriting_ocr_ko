import os
from src.data.stream import StreamingDataLoader
from src.data.load_label import *

def infer_step(imgs_batch):
    print("Test!")

if __name__ == "__main__":
    label_files = [os.path.splitext(f)[0] for f in os.listdir("data/test/labels") if f.endswith(".json")]

    infer_loader = StreamingDataLoader(label_files, batch_size=4, prefetch_batches=2, use_meta=False)
    try:
        for imgs_batch, metas_batch in infer_loader:
            print("Infer batch:", imgs_batch.shape, "Meta:", metas_batch)
            infer_step(imgs_batch)
    finally:
        infer_loader.shutdown()
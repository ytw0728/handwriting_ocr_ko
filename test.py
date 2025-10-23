import os
from tqdm import tqdm
from src.data.stream import StreamingDataLoader
from src.data.load_label import *
from src.models.deep_conv_net import DeepConvNet
from src.optimizers.adam import Adam
from src.training.output_encoder import *
import src.numpy as np

if __name__ == "__main__":
    label_files = [os.path.splitext(f)[0] for f in os.listdir("data/test/labels") if f.endswith(".json")]

    max_answer_length = 10
    infer_loader = StreamingDataLoader(label_files, batch_size=1, split='test', prefetch_batches=2, use_meta=False, 
    include_hanja=False, output_size=output_size, max_answer_length=max_answer_length)
    
    network = DeepConvNet(input_dim=(3,128,256),output_size=output_size * max_answer_length)
    optimizer = Adam(lr=0.01)
    train_acc_list, test_acc_list = [], []

    print("글자 가짓수", output_size)
    print("run by ", np.__name__, np.__version__)
    network.load_params("deep_conv_net_params.pkl")
    max_epochs = 10
    for epoch in tqdm(range(max_epochs)):
        infer_loader.trigger()
        try:
            for imgs_batch, _, answer_batch in infer_loader:
                print("Test batch:", imgs_batch.dtype, imgs_batch.shape)

                train_acc = network.accuracy(imgs_batch, answer_batch)
                test_acc = network.accuracy(imgs_batch, answer_batch)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("Train Acc:", train_acc, "Test Acc:", test_acc)

        finally:
            infer_loader.shutdown()
import os
from tqdm import tqdm
from src.data.stream import StreamingDataLoader
from src.data.load_label import *
from src.models.deep_conv_net import DeepConvNet
from src.optimizers.adam import Adam
from src.training.output_encoder import *

if __name__ == "__main__":
    label_files = [os.path.splitext(f)[0] for f in os.listdir("data/train/labels") if f.endswith(".json")]

    #max_file_size = 100000
    max_file_size = 1
    #max_epochs = 10
    max_epochs = 1
    max_answer_length = 10
    
    train_loader = StreamingDataLoader(label_files[0:max_file_size], batch_size=1, prefetch_batches=2, use_meta=True, 
    include_hanja=False, output_size=output_size, max_answer_length=max_answer_length)

    network = DeepConvNet(input_dim=(3,128,256),output_size=output_size * max_answer_length)
    optimizer = Adam(lr=0.01)
    train_loss_list, train_acc_list, test_acc_list = [], [], []

    print("글자 가짓수", output_size)
    print("max answer length", max_answer_length)
    print("max file size", max_file_size)
    print("max epochs", max_epochs)
    for epoch in tqdm(range(max_epochs)):
        train_loader.trigger()
        try:
            for imgs_batch, metas_batch, answer_batch in train_loader:
                print("Train batch:", imgs_batch.dtype, imgs_batch.shape, "Meta count:", len(metas_batch))

                grads, loss = network.gradient(imgs_batch, answer_batch)
                optimizer.update(network.params, grads)
                train_loss_list.append(loss)

                train_acc = network.accuracy(imgs_batch, answer_batch)
                test_acc = network.accuracy(imgs_batch, answer_batch)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

        finally:
            train_loader.shutdown()
    network.save_params("deep_conv_net_params.pkl")
    print("Training Done")
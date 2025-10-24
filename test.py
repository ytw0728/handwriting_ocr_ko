import os
from tqdm import tqdm
from src.data.stream import StreamingDataLoader
from src.data.load_label import *
# from src.models.deep_conv_net import DeepConvNet
from src.layers.ctc_loss import CTCLoss
from src.data.output_encoder import *
from src.models.vit import ViT
from src.optimizers.optimizer import Adam
from src import numpy as np
from src.data.output_encoder import output_size

if __name__ == "__main__":
    label_files = [os.path.splitext(f)[0] for f in os.listdir("data/test/labels") if f.endswith(".json")]

    # max_file_size = 100000
    max_file_size = 100
    max_epochs = 10
    # max_epochs = 1
    max_answer_length = 16

    infer_loader = StreamingDataLoader(label_files[0:max_file_size], split='test', batch_size=1, prefetch_batches=2, use_meta=True, 
    include_hanja=False, output_size=output_size, max_answer_length=max_answer_length)
    network = ViT((3,128,256), (8, 16), 128, 8, 4, 3688)
    loss_fn = CTCLoss(blank_idx=output_size)
    optimizer = Adam(learning_rate=0.00001)
    network.set_optimizer(optimizer)
    # print('model done')
    # network = DeepConvNet(input_dim=(3,128,256),output_size=output_size * max_answer_length)
    train_loss_list, train_acc_list, test_acc_list = [], [], []

    print("글자 가짓수", output_size)
    print("max answer length", max_answer_length)
    print("max file size", max_file_size)
    print("max epochs", max_epochs)
    print("run by ", np.__name__)
    
    outer_tqdm = tqdm(range(max_epochs), desc="Training Epochs")

    network.load_weights("vit_weights.pkl")

    for epoch in outer_tqdm:
        epoch_losses = []
        epoch_accs = []
        infer_loader.trigger()
        try:
            inner_tqdm = tqdm(infer_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
            for imgs_batch, answer_batch, answer_length_batch in inner_tqdm:
                # for imgs_batch, metas_batch, answer_batch in tqdm(infer_loader):
                # print("Train batch:", imgs_batch.dtype, imgs_batch.shape, answer_batch.shape)
                y_hat = network.forward(imgs_batch)
                loss = loss_fn.forward(y_hat, answer_batch, input_lengths=np.array([y_hat.shape[1]]*y_hat.shape[0]), target_lengths=answer_length_batch)

                train_acc = network.accuracy(imgs_batch, answer_batch, answer_length_batch)
                test_acc = network.accuracy(imgs_batch, answer_batch, answer_length_batch)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                inner_tqdm.set_postfix(loss=round(float(loss), 2), acc=round(float(train_acc), 2))

            avg_loss = float(sum(epoch_losses)) / max(1, len(epoch_losses))
            avg_acc = float(sum(epoch_accs)) / max(1, len(epoch_accs))
            outer_tqdm.set_postfix(loss=loss, acc=train_acc)
        finally:
            infer_loader.shutdown()
    print("Test Done")

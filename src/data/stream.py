import threading
import queue
import src.numpy as np
from .load_label import load_label
from .load_source import load_source

class StreamingDataLoader:
    def __init__(
        self,
        file_list: list[str],
        output_size:int,
        split:str='train',
        batch_size:int=8,
        resize:tuple[int,int]=(256,128),
        prefetch_batches:int=2,
    ) -> None:
        self.file_list = file_list
        self.split = split
        self.batch_size = batch_size
        self.resize = resize
        self.queue: queue.Queue[tuple[
            np.ndarray[np.float32, np.dtype[np.float32]],
            np.ndarray[np.ndarray[int]],
            np.ndarray[np.ndarray[int]],
        ]] = queue.Queue(maxsize=prefetch_batches)
        self._stop = threading.Event()
        self.output_size = output_size

    def _loader_worker(self) -> None:
        idx = 0

        imgs_batch: list[np.ndarray[np.float64, np.dtype[np.float64]]] = []
        answer_batch: list[np.array[int]] = []
        answer_length_batch: list[np.array[int]] = []

        while idx < len(self.file_list) and not self._stop.is_set():
            file = self.file_list[idx]

            label = load_label(self.split, file)
            if label is None:
                continue
            imgs, answers, answer_lengths = load_source(self.split, label, self.resize)

            imgs_batch.extend(imgs)
            answer_batch.extend(answers)
            answer_length_batch.extend(answer_lengths)

            if len(imgs_batch) >= self.batch_size  or idx == len(self.file_list) -1:
                imgs_array = np.array(imgs_batch[0:self.batch_size], dtype=np.float32)
                answer_array = np.array(answer_batch[0:self.batch_size], dtype=np.int32)
                answer_length_array = np.array(answer_length_batch[0:self.batch_size], dtype=np.int32)

                self.queue.put((imgs_array, answer_array, answer_length_array))

                imgs_batch = imgs_batch[self.batch_size:]
                answer_batch = answer_batch[self.batch_size:]
                answer_length_batch = answer_length_batch[self.batch_size:]

            idx += 1

        self.queue.put(None)

    def __iter__(self) -> 'StreamingDataLoader':
        return self

    def __next__(self) :
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch

    def __len__(self) -> int:
        return (len(self.file_list) + self.batch_size - 1) // self.batch_size

    def shutdown(self) -> None:
        self._stop.set()
        self.thread.join(timeout=1)
        while not self.queue.empty():
            self.queue.get()

    def trigger(self) -> None:
        self._stop.clear()

        np.random.shuffle(self.file_list)
        self.thread = threading.Thread(target=self._loader_worker)
        self.thread.daemon = True
        self.thread.start()
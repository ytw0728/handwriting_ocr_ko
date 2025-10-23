import threading
import queue
import numpy as np
from .load_label import load_label
from .load_source import load_source

class StreamingDataLoader:
    def __init__(
        self,
        file_list: list[str],
        output_size:int,
        max_answer_length:int,
        split:str='train',
        batch_size:int=8,
        resize:tuple[int,int]=(256,128),
        prefetch_batches:int=2,
        use_meta:bool=True,
        include_hanja:bool=False,
    ) -> None:
        self.file_list = file_list
        self.split = split
        self.batch_size = batch_size
        self.resize = resize
        self.use_meta = use_meta
        self.include_hanja = include_hanja
        self.queue: queue.Queue[tuple[
            np.ndarray[np.float32, np.dtype[np.float32]],
            np.ndarray[dict],
            np.ndarray[np.ndarray[int]],
        ]] = queue.Queue(maxsize=prefetch_batches)

        self._stop = threading.Event()
        
        self.output_size = output_size
        self.max_answer_length = max_answer_length

    def _loader_worker(self) -> None:
        idx = 0
        while idx < len(self.file_list) and not self._stop.is_set():
            batch_files = self.file_list[idx: idx+self.batch_size]

            imgs_batch: list[np.ndarray[np.float64, np.dtype[np.float64]]] = []
            metas_batch: list[dict] = []
            answer_batch: list[np.array[int]] = []

            for f in batch_files:
                label = load_label(self.split, f)
                if label is None:
                    continue
                imgs, metas, answers = load_source(self.split, label, self.output_size, self.max_answer_length, self.resize, self.include_hanja)
                
                imgs_batch.extend(imgs)
                metas_batch.extend(metas)
                answer_batch.extend(answers)
            if len(imgs_batch) > 0:
                if not self.use_meta:
                    metas_batch = None  # meta branch 사용 안함
                self.queue.put((
                    np.array(imgs_batch, dtype=np.float32),
                    np.array(metas_batch),
                    np.array(answer_batch),
                ))
            idx += self.batch_size

        self.queue.put(None)

    def __iter__(self) -> 'StreamingDataLoader':
        return self

    def __next__(self) -> tuple[np.ndarray[np.float32, np.dtype[np.float32]], np.ndarray[dict], np.ndarray[np.ndarray[int]]]:
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

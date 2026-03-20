from gabor_like._cosine_similarity import _cosine_similarity
from gabor_like._data import (
    _create_dataloader,
    _save_result,
    _save_result_labels,
    _prepare_output_folder,
    _save_labels,
)
from gabor_like._filterbank import _create_filterbank
import torch
import threading
import os


class CoeffPreProcessor:
    def __init__(
        self,
        dataset: str,
        img_res: int,
        sigma: float,
        batch_size: int,
        steps: int,
        percent: int,
    ):
        self.dataset = dataset
        self.img_res = img_res
        self.sigma = sigma
        self.batch_size = batch_size
        self.steps = steps
        self.percent = percent

        self.cv = threading.Condition()

        # Select device to run tensors on: CPU or CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_to_disk(self, output_path: str):
        self.output_path = os.path.join(
            output_path, f"coeff_{self.img_res}_{self.steps}_{self.percent}"
        )
        _prepare_output_folder(output_path)

        self._setup()

        # Get a DataLoader object and information about the dataset
        self.loader, classes = _create_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            img_res=self.img_res,
        )

        _save_labels(classes, output_path)

        # Record GPU memory for performance testing
        # if self.device.type == "cuda":
        #     torch.cuda.memory._record_memory_history()

        # Create consumer for writing results
        consumer = threading.Thread(target=self._consumer, daemon=True)
        consumer.start()

        # Start producing extracted features
        self._producer(self.device)

        # Wait for consumer to finish writing
        consumer.join()

        # if self.device.type == "cuda":
        #     torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    def _setup(self):
        self.work_available = False
        self.done = False
        self.results = None
        self.result_labels = None

        # Calculate kernel/filter size, needs to be an odd number close to image size
        # Calculate padding necessary for unfolding
        if self.img_res % 2 == 0:
            kernel_size = self.img_res - 1
        else:
            kernel_size = self.img_res
        padding = (int)((kernel_size - 1) / 2)

        # Get the filters as tensors
        self.filters_normalized, self.abs_filters = _create_filterbank(
            N=kernel_size,
            sigma=self.sigma,
            device=self.device,
            steps=self.steps,
            perc=self.percent,
        )

        # Create image unfolder
        self.unfold = torch.nn.Unfold(
            kernel_size=(kernel_size, kernel_size), padding=padding
        )

    def _consumer(self):
        file_index = 0

        while True:
            with self.cv:
                while not self.work_available and not self.done:
                    self.cv.wait()

                if not self.work_available and self.done:
                    break

                _save_result(self.results, file_index, self.output_path)
                _save_result_labels(self.result_labels, file_index, self.output_path)

                self.work_available = False
                print(file_index)
                file_index += 1

                self.cv.notify_all()

    def _producer(self, device):
        for images, labels in self.loader:
            images = images.to(device)

            # Extract sliding blocks from batched images
            unfolded_img = self.unfold(images)

            # Calculate cosine_similarity between unfolded image and filters, result is extracted features
            coeff = _cosine_similarity(
                self.filters_normalized, unfolded_img, self.abs_filters
            )

            with self.cv:
                while self.work_available:
                    self.cv.wait()
                self.results = coeff.to("cpu", non_blocking=True).clone()
                self.result_labels = labels.clone()
                self.work_available = True
                self.cv.notify_all()
            del coeff, unfolded_img

        with self.cv:
            self.done = True
            self.cv.notify_all()

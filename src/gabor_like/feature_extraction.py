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


class FeatureExtractor:
    def __init__(
        self,
        dataset: str,
        img_res: int,
        sigma: float,
        k: int,
        batch_size: int,
    ):
        self.dataset = dataset
        self.img_res = img_res
        self.sigma = sigma
        self.k = k
        self.batch_size = batch_size

        self.cv = threading.Condition()

        # Select device to run tensors on: CPU or CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_to_disk(self, output_path: str):
        self.output_path = output_path
        _prepare_output_folder(output_path)

        self._setup()

        # Get a DataLoader object and information about the dataset
        self.loader, classes, samples_len = _create_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            img_res=self.img_res,
        )

        _save_labels(classes, output_path)

        # Record GPU memory for performance testing
        # if device.type == "cuda":
        #    torch.cuda.memory._record_memory_history()

        # Create consumer for writing results
        consumer = threading.Thread(target=self._consumer_disk, daemon=True)
        consumer.start()

        # Start producing extracted features
        self._producer(self.device)

        # Wait for consumer to finish writing
        consumer.join()

        # if device.type == "cuda":
        #   torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    def extract_to_tensor(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        self._setup()

        # Get a DataLoader object and information about the dataset
        self.loader, classes, samples_len = _create_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            img_res=self.img_res,
        )

        self.results_tensor = torch.empty(
            (
                samples_len,
                self.filters_normalized.size(dim=0) * 2,
                self.img_res,
                self.img_res,
            )
        )
        self.result_label_tensor = torch.empty(samples_len, dtype=torch.long)

        # Record GPU memory for performance testing
        # if device.type == "cuda":
        #    torch.cuda.memory._record_memory_history()

        # Create consumer for writing results
        consumer = threading.Thread(target=self._consumer_tensor, daemon=True)
        consumer.start()

        # Start producing extracted features
        self._producer(self.device)

        # Wait for consumer to finish writing
        consumer.join()

        # if device.type == "cuda":
        #   torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

        return self.results_tensor, self.result_label_tensor, classes

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
        self.filters_normalized, self.abs_filters, self.labels = _create_filterbank(
            N=kernel_size, sigma=self.sigma, k=self.k, device=self.device
        )

        # Create image unfolder
        self.unfold = torch.nn.Unfold(
            kernel_size=(kernel_size, kernel_size), padding=padding
        )

    def _consumer_disk(self):
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

    def _consumer_tensor(self):
        batch_index = 0
        write_offset = 0

        while True:
            with self.cv:
                print()
                while not self.work_available and not self.done:
                    self.cv.wait()

                if not self.work_available and self.done:
                    break

                batch_size = self.results.size(dim=0)
                self.results_tensor[write_offset : write_offset + batch_size] = (
                    self.results
                )
                self.result_label_tensor[write_offset : write_offset + batch_size] = (
                    self.result_labels
                )
                write_offset += batch_size

                self.work_available = False
                print(batch_index)
                batch_index += 1

                self.cv.notify_all()

    def _producer(self, device):
        for images, labels in self.loader:
            images = images.to(device)

            # Extract sliding blocks from batched images
            unfolded_img = self.unfold(images)

            # Calculate cosine_similarity between unfolded image and filters, result is extracted features
            features = _cosine_similarity(
                self.filters_normalized, unfolded_img, self.abs_filters
            )

            filter_amount = self.filters_normalized.size(dim=0) * 2
            current_batch_size = features.size(dim=0)

            max_coeff_list = []
            for i in range(self.k):
                for j in range(4):
                    sample = features[:, j::4]
                    coeff_in_cluster = sample[:, labels == i]
                    if coeff_in_cluster.size(dim=1) == 0:
                        continue
                    max_coeff = torch.amax(coeff_in_cluster, dim=1)
                    max_coeff_list.append(max_coeff)

            features = torch.stack(max_coeff_list)

            features = features.view(
                current_batch_size, self.k*4, self.img_res, self.img_res
            )
            with self.cv:
                while self.work_available:
                    self.cv.wait()
                self.results = features.to("cpu", non_blocking=True).clone()
                self.result_labels = labels.clone()
                self.work_available = True
                self.cv.notify_all()
            del unfolded_img, features
        with self.cv:
            self.done = True
            self.cv.notify_all()

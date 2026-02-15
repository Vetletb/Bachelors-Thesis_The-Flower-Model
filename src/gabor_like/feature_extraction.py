from gabor_like._cosine_similarity import _cosine_similarity
from gabor_like._data import _create_dataloader, _save_result, _save_result_labels
from gabor_like._filterbank import _create_filterbank
import torch
import threading


class FeatureExtractor:
    def __init__(self, dataset: str, output: str, img_res: int, sigma: float, batch_size: int):
        self.dataset = dataset
        self.output = output
        self.img_res = img_res
        self.sigma = sigma
        self.batch_size = batch_size

        self.cv = threading.Condition()

    def run(self):
        self.work_available = False
        self.done = False
        self.results = None
        self.result_labels = None

        # Select device to run tensors in: CPU or CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Record GPU memory for performance testing
        if device.type == "cuda":
            torch.cuda.memory._record_memory_history()

        # Calculate kernel/filter size, needs to be an odd number close to image size
        # Calculate padding necessary for unfolding
        if self.img_res % 2 == 0:
            kernel_size = self.img_res - 1
        else:
            kernel_size = self.img_res
        padding = (int)((kernel_size - 1) / 2)

        # Get the filters as tensors
        self.filters_normalized, self.frequency_domain_sum = _create_filterbank(
            N=kernel_size, sigma=self.sigma, device=device
        )

        # Create image unfolder
        self.unfold = torch.nn.Unfold(
            kernel_size=(kernel_size, kernel_size), padding=padding
        )

        # Get a DataLoader object and iterate through the batches
        self.loader = _create_dataloader(
            dataset=self.dataset, output=self.output, batch_size=self.batch_size, img_res=self.img_res
        )

        # Create consumer for writing results
        consumer = threading.Thread(target=self._consumer, daemon=True)
        consumer.start()

        # Start producing extracted features
        self._producer(device)

        # Wait for consumer to finish writing
        consumer.join()

        if device.type == "cuda":
            torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    def _consumer(self):
        file_index = 0

        while True:
            with self.cv:
                print()
                while not self.work_available and not self.done:
                    self.cv.wait()

                if not self.work_available and self.done:
                    break

                # save_result(results, file_index)
                # save_result_labels(result_labels, file_index)

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
            features = _cosine_similarity(
                self.filters_normalized, unfolded_img, self.frequency_domain_sum
            )

            filter_amount = self.filters_normalized.size(dim=0)
            current_batch_size = features.size(dim=0)
            features = features.view(
                current_batch_size, filter_amount, self.img_res, self.img_res
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

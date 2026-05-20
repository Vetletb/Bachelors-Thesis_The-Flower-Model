from ._tensor_ops import _cosine_similarity
from ._data import (
    _create_dataloader,
    _save_train_result,
    _save_train_result_labels,
    _save_test_result,
    _save_test_result_labels,
    _prepare_output_folder,
    _save_labels,
)
from ._filterbank import _create_filterbank
import torch
import threading
import os


class CoeffPreProcessor:
    """
    Preprocesses the dataset by using a predefined filterbank. Saves all coefficients extracted from the dataset to disk.

    Args:
        dataset: Path of dataset folder.
        img_res: The resolution the images gets scaled to.
        sigma: Sigma used for generating filters.
        steps: How many filter sizes of same rotation gets generated.
        percent: Determines minimum filter size. Between 0-1.
        batch_size: Batch size of the images being processed at a time.

    Notes:
        Dataset needs to be structured like this (images does not need to be .png):
        root/classname/image.png
    """

    def __init__(
        self,
        dataset: str,
        img_res: int,
        sigma: float,
        steps: int,
        percent: float,
        batch_size: int,
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

    def preprocess(self, output_path: str):
        """
        Preprocesses the dataset by using a predefined filterbank.
        Saves all coefficients extracted from the dataset to disk.
        Automatically separates the dataset to train and test set with 80/20 split.

        Args:
            output_path: Folder for storing the coefficients, labels and classnames.

        Notes:
            Recommended to use same output folder for same dataset, but different parameters.
            Different datasets should have different output folders.
        """
        self.output_path = os.path.join(
            output_path,
            f"coeff_{self.img_res}_{self.sigma}_{self.steps}_{self.percent}",
        )
        _prepare_output_folder(self.output_path)

        self._setup()

        # Record GPU memory for performance testing
        # if self.device.type == "cuda":
        #     torch.cuda.memory._record_memory_history()

        # Create consumer for writing results
        consumer = threading.Thread(target=self._consumer, daemon=True)
        consumer.start()

        # Start producing extracted coefficients
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
            percent=self.percent,
        )

        # Create image unfolder
        self.unfold = torch.nn.Unfold(
            kernel_size=(kernel_size, kernel_size), padding=padding
        )

        # Get a DataLoader object and information about the dataset
        self.trainloader, self.testloader, classes = _create_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            img_res=self.img_res,
        )

        _save_labels(classes, self.output_path)

    def _consumer(self):
        """
        Write coefficient batches to disk sequentially.

        Waits for producer to set work_available, saves self.results and
        self.result_labels, signals producer to continue. Exits when producer
        signals self.done.
        """
        train_file_index = 0
        test_file_index = 0

        while True:
            with self.cv:
                while not self.work_available and not self.done:
                    self.cv.wait()

                if not self.work_available and self.done:
                    break

                if self.loader_idx == 0:
                    _save_train_result(self.results, train_file_index, self.output_path)
                    _save_train_result_labels(
                        self.result_labels, train_file_index, self.output_path
                    )
                    train_file_index += 1
                    print(train_file_index)

                else:
                    _save_test_result(self.results, test_file_index, self.output_path)
                    _save_test_result_labels(
                        self.result_labels, test_file_index, self.output_path
                    )
                    test_file_index += 1
                    print(test_file_index)

                self.work_available = False

                self.cv.notify_all()

    def _producer(self, device):
        """
        Extracts coefficients for batches and makes them available for consumer.

        Iterates dataloader, unfolds images into patches, computes cosine similarity
        with normalized filters, and enqueues results for consumer thread.
        Synchronizes via self.cv to avoid overwriting unsaved batches.

        Args:
            device: Torch device for tensor placement
        """
        self.loader_idx = 0
        for loader in [self.trainloader, self.testloader]:
            for images, labels in loader:
                images = images.to(device)

                # Extract sliding blocks from batched images
                unfolded_img = self.unfold(images)

                # Calculate cosine similarity between unfolded image and filters, result is extracted coefficients
                coeff = _cosine_similarity(
                    self.filters_normalized, unfolded_img, self.abs_filters
                )

                # Waits for consumer, then makes current batch available
                with self.cv:
                    while self.work_available:
                        self.cv.wait()
                    self.results = coeff.cpu().clone()
                    self.result_labels = labels.clone()
                    self.work_available = True
                    self.cv.notify_all()
                del coeff, unfolded_img

            # Waits for consumer to finish before changing loader
            with self.cv:
                while self.work_available:
                    self.cv.wait()
                self.loader_idx += 1

        with self.cv:
            self.done = True
            self.cv.notify_all()

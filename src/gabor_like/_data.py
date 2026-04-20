from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import os

PREPROCESSED_FOLDER = "preprocessed_coeff"
LABEL_RESULTS_FOLDER = "labels"


def _prepare_output_folder(path: str):
    """
    Create output directory tree for extracted features and labels.

    Expected structure:
    - {path}/preprocessed_coeff/
    - {path}/labels/

    Args:
        path: Root output directory.
    """
    preprocessed_path = os.path.join(path, PREPROCESSED_FOLDER)
    labels_path = os.path.join(path, LABEL_RESULTS_FOLDER)
    os.makedirs(path)
    os.makedirs(preprocessed_path)
    os.makedirs(labels_path)


def _prepare_folder(path: str):
    """
    Create a directory.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path)


def _create_dataloader(
    dataset: str, batch_size: int, img_res: int
) -> tuple[torch.utils.data.DataLoader, list[str]]:
    """
    Build a dataloader of scaled grayscale images.

    Args:
        dataset: Root folder of image dataset. Dataset needs to be structured 
        like this (images does not need to be .png):
        root/classname/image.png
        batch_size: Number of samples per batch.
        img_res: Target width and height in pixels.

    Returns:
        loader: DataLoader with shuffle disabled and pin_memory enabled.
        classes: List of class names in index order.
    """
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_res, img_res)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(root=dataset, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return loader, dataset.classes


def _save_result(result: torch.Tensor, index: int, path: str):
    """
    Save result tensor for a single batch.

    Args:
        result: tensor to save.
        index: batch index used as filename stem.
        path: Root output directory.
    """
    features_path = os.path.join(path, PREPROCESSED_FOLDER, f"{index}.pt")
    torch.save(result, features_path)


def _save_result_labels(labels: torch.Tensor, index: int, path: str):
    """
    Save label tensor for a single batch.

    Args:
        labels: Label tensor to save.
        index: batch index used as filename stem.
        path: Root output directory.
    """
    label_results_path = os.path.join(path, LABEL_RESULTS_FOLDER, f"{index}.pt")
    torch.save(labels, label_results_path)


def _save_labels(labels: list[str], path: str):
    """
    Save classnames.

    Args:
        labels: Ordered class names where position equals class index.
        path: Root output directory.
    """
    labels_path = os.path.join(path, "classes.pt")
    torch.save(labels, labels_path)


def _save_pooled(result: torch.Tensor, index: int, path: str):
    """
    Save pooled tensor output for a single batch.

    Args:
        result: Pooled tensor to save.
        index: batch index used as filename stem.
        path: Destination directory.
    """
    pool_path = os.path.join(path, f"{index}.pt")
    torch.save(result, pool_path)

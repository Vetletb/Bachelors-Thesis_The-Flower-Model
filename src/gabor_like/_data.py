from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import os

PREPROCESSEDS_FOLDER = "preprocessed_coeff"
LABEL_RESULTS_FOLDER = "labels"


def _prepare_output_folder(path: str):
    preprocessed_path = os.path.join(path, PREPROCESSEDS_FOLDER)
    labels_path = os.path.join(path, LABEL_RESULTS_FOLDER)
    os.makedirs(path)
    os.makedirs(preprocessed_path)
    os.makedirs(labels_path)


def _prepare_folder(path: str):
    os.makedirs(path)


def _create_dataloader(
    dataset: str, batch_size: int, img_res: int
) -> tuple[torch.utils.data.DataLoader, list[str]]:
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
    features_path = os.path.join(path, PREPROCESSEDS_FOLDER, f"{index}.pt")
    torch.save(result, features_path)


def _save_result_labels(labels: torch.Tensor, index: int, path: str):
    label_results_path = os.path.join(path, LABEL_RESULTS_FOLDER, f"{index}.pt")
    torch.save(labels, label_results_path)


def _save_labels(labels: list[str], path: str):
    labels_path = os.path.join(path, "classes.pt")
    torch.save(labels, labels_path)


def _save_pooled(result: torch.Tensor, index: int, path: str):
    pool_path = os.path.join(path, f"{index}.pt")
    torch.save(result, pool_path)

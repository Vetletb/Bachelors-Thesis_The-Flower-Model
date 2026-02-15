from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import os

FEATURES_FOLDER = "extracted_features"
LABEL_RESULTS_FOLDER = "labels"

def _prepare_output_folder(path: str):
    features_path = os.path.join(path, FEATURES_FOLDER)
    labels_path = os.path.join(path, LABEL_RESULTS_FOLDER)
    os.makedirs(features_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)


def _create_dataloader(
    dataset: str, output: str, batch_size: int, img_res: int
) -> torch.utils.data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_res, img_res)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(root=dataset, transform=transform)

    _save_labels(dataset.classes, output)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return loader


def _save_result(result: torch.Tensor, index: int, path: str):
    features_path = os.path.join(path, FEATURES_FOLDER, f"{index}.pt")
    torch.save(result, features_path)


def _save_result_labels(labels: torch.Tensor, index: int, path: str):
    label_results_path = os.path.join(path, LABEL_RESULTS_FOLDER, f"{index}.pt")
    torch.save(labels, label_results_path)


def _save_labels(labels: list[str], path: str):
    labels_path = os.path.join(path, "classes.pt")
    torch.save(labels, labels_path)

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch


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

    # _save_labels(dataset.classes, output)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return loader


def _save_result(result: torch.Tensor, index: int, path: str):
    torch.save(result, "output/extracted_features/" + str(index) + ".pt")


def _save_result_labels(labels: torch.Tensor, index: int, path: str):
    torch.save(labels, "output/labels/" + str(index) + ".pt")


def _save_labels(labels: list[str], path: str):
    torch.save(labels, "output/classes.pt")

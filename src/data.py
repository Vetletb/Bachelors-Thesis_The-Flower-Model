from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch


def create_dataloader(
    path: str, batch_size: int, img_res: int
) -> torch.utils.data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_res, img_res)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(root=path, transform=transform)

    save_labels(dataset.classes)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return loader


def save_result(result: torch.Tensor, index: int):
    torch.save(result, "output/extracted_features/" + str(index) + ".pt")


def save_result_labels(labels: torch.Tensor, index: int):
    torch.save(labels, "output/labels/" + str(index) + ".pt")


def save_labels(labels: list[str]):
    torch.save(labels, "output/classes.pt")

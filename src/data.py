from torchvision import transforms 
from torchvision.datasets import ImageFolder
import torch

def get_dataloader(path: str, batch_size: int, img_res: int) -> torch.utils.data.DataLoader:
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_res, img_res)),
    transforms.ToTensor()
    ])
    dataset = ImageFolder(root=path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader
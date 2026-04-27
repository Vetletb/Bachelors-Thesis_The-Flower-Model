import torch
from torchvision.transforms import v2


def _mix_augment_coeff(
    coeffs: torch.Tensor, aug_amount: int, img_res: int
) -> torch.Tensor:
    """
    Augments the coeff tensor with a mix of both translates and scaling.
    Input tensor needs to be of shape (images, filters, coeffs). Augmentations are random based on a normal distribution

    Args:
        coeffs: The input tensor of shape (images, filters, coeffs).
        aug_amount: How many times the images gets augmented.
        img_res: Resolution of the images.

    Returns:
        augmented: new tensor of shape (images, aug_amount ,filters, coeffs).
        Adds a new dimention for the augmented images.
    """
    image_amount, filter_amount, _ = coeffs.size()
    augmented = coeffs.view(image_amount, 1, filter_amount, img_res, img_res)
    augmented = augmented.repeat(1, aug_amount, 1, 1, 1)

    translate = torch.normal(
        0.0, img_res * 0.2, (aug_amount * image_amount, 2)
    ).tolist()
    scale = (
        torch.normal(1.0, 0.25, (aug_amount * image_amount,)).clamp(min=0.1).tolist()
    )

    for i in range(image_amount):
        for j in range(aug_amount):
            augmented[i, j] = v2.functional.affine(
                augmented[i, j],
                translate=translate[i * aug_amount + j],
                scale=scale[i * aug_amount + j],
                angle=0.0,
                shear=[0.0, 0.0],
            )

    augmented = v2.RandomHorizontalFlip()(augmented)

    return augmented


def _augment_coeff(coeffs: torch.Tensor, aug_amount: int, img_res: int):
    """
    Augments the coeff tensor by flipping horizontal once, then seperate translate and scale `aug_amount` times.
    Input tensor needs to be of shape (images, filters, coeffs). Augmentations are random based on a normal distribution.
    First augmentation is the original images, followed by all flipped, then translated and scaled

    Args:
        coeffs: The input tensor of shape (images, filters, coeffs).
        aug_amount: How many times the images gets augmented.
        img_res: Resolution of the images.

    Returns:
        augmented: new tensor of shape (images, aug_amount ,filters, coeffs).
        Adds a new dimention for the augmented images.
    """
    image_amount, filter_amount, _ = coeffs.size()
    augmented = coeffs.view(image_amount, 1, filter_amount, img_res, img_res)
    augmented = augmented.repeat(1, aug_amount * 2 + 2, 1, 1, 1)

    translate = torch.normal(
        0.0, img_res * 0.2, size=(aug_amount * image_amount, 2)
    ).tolist()
    scale = (
        torch.normal(1.0, 0.2, size=(aug_amount * image_amount,))
        .clamp(min=0.1)
        .tolist()
    )

    augmented[:, 1] = v2.functional.horizontal_flip(augmented[:, 1])

    for i in range(image_amount):
        for j in range(aug_amount):
            augmented[i, j + 2] = v2.functional.affine(
                augmented[i, j + 2],
                translate=translate[i * aug_amount + j],
                scale=1.0,
                angle=0.0,
                shear=[0.0, 0.0],
            )
            augmented[i, j + aug_amount + 2] = v2.functional.affine(
                augmented[i, j + aug_amount + 2],
                translate=[0.0, 0.0],
                scale=scale[i * aug_amount + j],
                angle=0.0,
                shear=0.0,
            )

    return augmented

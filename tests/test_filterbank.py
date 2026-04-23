import pytest
import gabor_like._filterbank as fb
import torch


def test_shift_filters_returns_correct_value():
    a = torch.arange(1, 3 * 3 * 2 + 1).view(2, 3, 3)
    print(a)

    even = True
    offset = 1 if even else 0

    shifted, position = fb._shift_filters(a, 2, 3 + offset)

    print(shifted.shape)
    print(position.shape)

    expected_shifted_1 = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 0, 0, 0],
            [7, 8, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    expected_position_1 = torch.tensor([0, 0])

    expected_shifted_2 = torch.tensor(
        [
            [0, 1, 2, 3, 0, 0],
            [0, 4, 5, 6, 0, 0],
            [0, 7, 8, 9, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    expected_position_2 = torch.tensor([1, 0])

    expected_shifted_3 = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 10, 11, 12],
            [0, 0, 0, 13, 14, 15],
            [0, 0, 0, 16, 17, 18],
        ]
    )
    expected_position_3 = torch.tensor([3, 3])

    assert torch.allclose(expected_shifted_1, shifted[0], rtol=1e-5, atol=1e-8)
    assert torch.allclose(expected_position_1, position[0], rtol=1e-5, atol=1e-8)

    assert torch.allclose(expected_shifted_2, shifted[2], rtol=1e-5, atol=1e-8)
    assert torch.allclose(expected_position_2, position[2], rtol=1e-5, atol=1e-8)

    assert torch.allclose(expected_shifted_3, shifted[-1], rtol=1e-5, atol=1e-8)
    assert torch.allclose(expected_position_3, position[-1], rtol=1e-5, atol=1e-8)

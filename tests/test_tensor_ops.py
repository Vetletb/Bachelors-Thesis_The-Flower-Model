import pytest
import flower_model._tensor_ops as to
import torch


@pytest.mark.parametrize(
    "f, i, p",
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 63),
        (1, 0, 23),
        (1, 1, 14),
        (1, 1, 62),
    ],
)
def test_cosine_similarity_returns_correct_value(f, i, p):
    b = torch.arange(0, 8 * 8 * 2, dtype=torch.float32).view(2, 1, 8, 8)

    unfold = torch.nn.Unfold(kernel_size=(7, 7), padding=3)
    b = unfold(b)

    a = torch.arange(0, 7 * 7 * 2 * 2, dtype=torch.float32).view(4, 49)
    a_normalized = torch.nn.functional.normalize(a)
    a_abs = torch.arange(0, 0.1 * 7 * 7 * 2, 0.1, dtype=torch.float32).view(2, 49)

    result = to._cosine_similarity(a_normalized, b, a_abs)

    real_expected = torch.nn.functional.cosine_similarity(
        a[f], b[i, :, p] * a_abs[f], dim=0
    )
    imag_expected = torch.nn.functional.cosine_similarity(
        a[f + 2], b[i, :, p] * a_abs[f], dim=0
    )

    assert torch.allclose(real_expected, result[i, f * 2, p], rtol=1e-5, atol=1e-8)
    assert torch.allclose(imag_expected, result[i, f * 2 + 1, p], rtol=1e-5, atol=1e-8)


def test_normalize_in_place_returns_correct_value():
    a = torch.rand((5, 4, 10), dtype=torch.float32)
    expected = torch.nn.functional.normalize(a, dim=2)
    to._normalize_in_place(a, dim=2)

    assert torch.allclose(expected, a, rtol=1e-5, atol=1e-8)


def test_normalize_in_place_divide_by_zero_does_not_return_not_a_number():
    a = torch.zeros((5, 4, 10), dtype=torch.float32)

    to._normalize_in_place(a, dim=2)

    assert torch.isfinite(a).all()
    assert not torch.isnan(a).any()

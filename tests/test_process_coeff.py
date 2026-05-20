import pytest
import flower_model
import torch
import os


@pytest.mark.parametrize("i", [0, 1, 2, 3, 4, 5])
def test_cluster_sets_correct_order(i, monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda path: ["a"])
    processor = flower_model.CoeffProcessor("path", 4, 1.0, 2, 1.0)
    processor.device = "cpu"

    processor.cluster(4)

    expected_position = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [3, 2],
            [0, 3],
            [1, 3],
            [2, 3],
            [3, 3],
        ],
        dtype=torch.float32,
    )

    actual_position = processor.filter_pos

    assert torch.allclose(
        expected_position, actual_position[i * 16 : (i + 1) * 16], rtol=1e-5, atol=1e-8
    )


def test_eye_labels_returns_correct_values(monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda path: ["a"])
    processor = flower_model.CoeffProcessor("path", 4, 1.0, 4, 1.0)
    processor.device = "cpu"

    processor.img_res = 4
    processor.low_freq_upper = 0.6
    processor.high_freq_lower = 0.8
    processor.N = 3
    processor.k_around_eye = 1
    processor.k = 5

    processor.filter_r = torch.tensor([0.25, 0.5, 0.75, 1.0]).repeat_interleave(4)
    processor.filter_pos = torch.tensor(
        [[2, 1], [1, 2], [2, 0], [3, 0]], dtype=torch.float32
    ).repeat(4, 1)

    processor.cluster_labels = torch.tensor(
        [4, 4, 3, 2, 4, 4, 3, 2, 1, 1, 3, 2, 2, 1, 0, 0]
    )

    result = processor._eye_labels()

    actual_low = torch.tensor([4, 3])
    actual_mid = torch.tensor([2])
    actual_high = torch.tensor([1, 0])

    actual_low_eye = actual_low[0]
    actual_mid_eye = actual_mid[0]
    actual_high_eye = actual_high[0]

    assert (result == actual_low_eye).any()
    assert (result == actual_mid_eye).any()
    assert (result == actual_high_eye).any()


def test_process_correctly_max_pools_coefficients(monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda path: ["a"])

    processor = flower_model.CoeffProcessor("path", 3, 1.0, 4, 1.0)
    processor.device = "cpu"
    processor.cluster_labels = torch.tensor(
        [0, 3, 2, 1, 2, 3, 2, 1, 0, 1, 4, 3, 2, 3, 4, 3, 2, 1]
    )
    processor.k = 4

    captured = {}

    def fake_save_pooled(max_coeff, i, pool_path):
        captured["max_coeff"] = max_coeff

    monkeypatch.setattr(
        flower_model.CoeffProcessor, "_eye_labels", lambda self: torch.tensor([0, 2, 3])
    )
    monkeypatch.setattr(
        flower_model.process_coeff, "_prepare_folder", lambda pool_path: None
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda path: torch.tensor(
            [
                [
                    [83, 67, -53, -68, 55, -36, -23, -28, 93],
                    [-33, -81, 33, -30, -51, -18, 12, 16, 92],
                    [75, -60, 38, -36, -19, 53, 55, 73, -70],
                    [27, -44, -28, -18, -23, 21, -24, 95, -59],
                ]
            ]
        ),
    )
    monkeypatch.setattr(flower_model.process_coeff, "_save_pooled", fake_save_pooled)

    processor.process(1, 1, 1, 0)

    assert (captured["max_coeff"] == 93).any()
    assert (captured["max_coeff"] == 95).any()
    assert (captured["max_coeff"] == 81).any()

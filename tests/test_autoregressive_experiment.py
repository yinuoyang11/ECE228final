import torch

from lerobot_policy_pi0_lite_flow.autoregressive_experiment import (
    masked_action_error_sums,
    split_by_episode,
)
from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset


class FakeColumnDataset:
    column_names = ["episode_index", "task_index", "action"]

    def __init__(self):
        self.rows = [
            {
                "episode_index": episode,
                "task_index": 20,
                "action": torch.zeros(7),
                "observation.images.image": torch.zeros(3, 8, 8),
                "observation.images.image2": torch.zeros(3, 8, 8),
                "observation.state": torch.zeros(8),
            }
            for episode in range(5)
            for _ in range(3)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]


def test_episode_split_has_no_overlap():
    dataset = LIBEROActionChunkDataset(
        base_dataset=FakeColumnDataset(),
        horizon=2,
        task_indices=[20],
        task_descriptions={20: "task 20"},
    )

    split = split_by_episode(dataset, eval_fraction=0.2, seed=42)

    assert set(split["train_episode_ids"]).isdisjoint(split["eval_episode_ids"])
    assert len(split["train_indices"]) + len(split["eval_indices"]) == len(dataset)


def test_masked_action_errors_ignore_padded_steps():
    target = torch.zeros(1, 3, 2)
    prediction = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
    action_is_pad = torch.tensor([[False, False, True]])

    mse_sum, l1_sum, count = masked_action_error_sums(prediction, target, action_is_pad)

    assert mse_sum.item() == 10.0
    assert l1_sum.item() == 6.0
    assert count.item() == 4

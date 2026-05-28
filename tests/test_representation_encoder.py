import torch

from lerobot_policy_pi0_lite_flow.configuration_pi0_lite_flow import PI0LiteFlowConfig
from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset, libero_samples_to_encoder_batch
from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import ACTION, COND_KEY, PI0LiteFlowPolicy
from lerobot_policy_pi0_lite_flow.representation_encoder import (
    RepresentationEncoder,
    RepresentationEncoderConfig,
    preprocess_clip_images,
)


def make_encoder_batch(batch_size: int = 3):
    return {
        "images": torch.randint(0, 256, (batch_size, 2, 3, 64, 64), dtype=torch.uint8),
        "state": torch.randn(batch_size, 8),
        "instruction": [f"pick object {idx}" for idx in range(batch_size)],
    }


def test_representation_encoder_outputs_decoder_condition_shape():
    encoder = RepresentationEncoder(RepresentationEncoderConfig(cond_dim=256, num_views=2))
    cond = encoder(make_encoder_batch())

    assert cond.shape == (3, 256)
    assert not torch.isnan(cond).any()


def test_representation_encoder_adds_condition_key_for_flow_policy():
    config = PI0LiteFlowConfig(
        horizon=16,
        action_dim=7,
        cond_dim=256,
        hidden_dim=64,
        num_layers=2,
        inference_steps=2,
    )
    policy = PI0LiteFlowPolicy(config)
    encoder = RepresentationEncoder(RepresentationEncoderConfig(cond_dim=config.cond_dim, num_views=2))

    batch = encoder.add_condition_to_batch(make_encoder_batch(batch_size=2))
    batch[ACTION] = torch.randn(2, config.horizon, config.action_dim)

    assert batch[COND_KEY].shape == (2, config.cond_dim)
    output = policy.forward(batch)
    assert torch.isfinite(output["loss"])


def test_libero_adapter_converts_hwc_images_to_encoder_batch():
    samples = [
        {
            "observation.images.image": torch.zeros(32, 32, 3, dtype=torch.uint8),
            "observation.images.image2": torch.ones(32, 32, 3, dtype=torch.uint8),
            "observation.state": torch.arange(8),
            "task": "move the bowl",
        },
        {
            "observation.images.image": torch.zeros(32, 32, 3, dtype=torch.uint8),
            "observation.images.image2": torch.ones(32, 32, 3, dtype=torch.uint8),
            "observation.state": torch.arange(8) + 1,
            "task": "open the drawer",
        },
    ]

    batch = libero_samples_to_encoder_batch(samples)

    assert batch["images"].shape == (2, 2, 3, 32, 32)
    assert batch["state"].shape == (2, 8)
    assert batch["instruction"] == ["move the bowl", "open the drawer"]


def test_preprocess_clip_images_resizes_and_normalizes():
    images = torch.randint(0, 256, (2, 3, 40, 50), dtype=torch.uint8)
    pixel_values = preprocess_clip_images(images)

    assert pixel_values.shape == (2, 3, 224, 224)
    assert pixel_values.dtype == torch.float32
    assert torch.isfinite(pixel_values).all()


class FakeEpisodeTable:
    def __iter__(self):
        return iter(
            [
                {"dataset_from_index": 0, "dataset_to_index": 3, "length": 3},
                {"dataset_from_index": 3, "dataset_to_index": 5, "length": 2},
            ]
        )


class FakeMeta:
    episodes = FakeEpisodeTable()


class FakeLIBERODataset:
    meta = FakeMeta()

    def __init__(self):
        self.rows = []
        for idx in range(5):
            episode_index = 0 if idx < 3 else 1
            self.rows.append(
                {
                    "observation.images.image": torch.zeros(3, 16, 16),
                    "observation.images.image2": torch.ones(3, 16, 16),
                    "observation.state": torch.full((8,), float(idx)),
                    "action": torch.full((7,), float(idx)),
                    "task": f"task {episode_index}",
                    "episode_index": torch.tensor(episode_index),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_action_chunk_dataset_does_not_cross_episode_boundary():
    dataset = LIBEROActionChunkDataset(base_dataset=FakeLIBERODataset(), horizon=4)

    item = dataset[1]

    assert item["images"].shape == (2, 3, 16, 16)
    assert item["state"].shape == (8,)
    assert item["instruction"] == "task 0"
    assert item["action"].shape == (4, 7)
    assert item["action_is_pad"].tolist() == [False, False, True, True]
    assert item["action"][:, 0].tolist() == [1.0, 2.0, 0.0, 0.0]


def test_action_chunk_dataset_handles_full_chunk_inside_episode():
    dataset = LIBEROActionChunkDataset(base_dataset=FakeLIBERODataset(), horizon=2)

    item = dataset[3]

    assert item["action_is_pad"].tolist() == [False, False]
    assert item["action"][:, 0].tolist() == [3.0, 4.0]

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.configuration_pi0_lite_flow import PI0LiteFlowConfig  # noqa: E402
from lerobot_policy_pi0_lite_flow.modeling_pi0_lite_flow import COND_KEY, PI0LiteFlowPolicy  # noqa: E402
from lerobot_policy_pi0_lite_flow.representation_encoder import (  # noqa: E402
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)


DEFAULT_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NO_OP_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless LIBERO rollout evaluation with MP4 recording.")
    parser.add_argument("checkpoint")
    parser.add_argument("--suite", default="libero_object")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-ids", nargs="+", type=int, help="Task ids local to the selected suite.")
    task_group.add_argument(
        "--dataset-task-indices",
        nargs="+",
        type=int,
        help="Global Hugging Face dataset task indices. These are mapped to suite-local ids by instruction text.",
    )
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None, help="Flow Euler inference steps.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-view", choices=("agentview", "wrist", "both"), default="agentview")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    set_seed(args.seed)
    device = torch.device(args.device)
    encoder, policy = load_model(Path(args.checkpoint), device, args.num_steps)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        raise ValueError(f"Unknown suite {args.suite!r}. Available suites: {sorted(benchmark_dict)}")
    suite = benchmark_dict[args.suite]()
    task_specs = resolve_task_specs(
        suite,
        task_ids=args.task_ids,
        dataset_task_indices=args.dataset_task_indices,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    max_steps = args.max_steps or DEFAULT_MAX_STEPS.get(args.suite, 500)
    video_dir = Path(args.video_dir) if args.video_dir else Path(args.checkpoint).parent / "rollout_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task_id, dataset_task_index in task_specs:
        task = suite.get_task(task_id)
        init_states = load_init_states(suite, task_id, get_libero_path("init_states"))
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        for episode in range(args.episodes_per_task):
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                camera_heights=args.image_size,
                camera_widths=args.image_size,
            )
            try:
                obs = reset_env(env, init_states[episode % len(init_states)], args.seed + episode, args.warmup_steps)
                policy.reset()
                frames = []
                success = False
                steps = 0
                for steps in range(1, max_steps + 1):
                    frames.append(make_video_frame(obs, args.video_view))
                    batch = observation_to_encoder_batch(obs, task.language, device)
                    with torch.no_grad():
                        cond = encoder(batch)
                        action = policy.select_action({COND_KEY: cond}, num_steps=policy.config.inference_steps)
                    obs, _, done, _ = env.step(action[0].detach().cpu().numpy())
                    success = bool(done or env.check_success())
                    if success:
                        frames.append(make_video_frame(obs, args.video_view))
                        break
            finally:
                env.close()

            outcome = "success" if success else "failure"
            video_path = video_dir / (
                f"{args.suite}_task{task_id:02d}_episode{episode:02d}_{outcome}.mp4"
            )
            write_mp4(video_path, frames, args.fps)
            result = {
                "suite": args.suite,
                "task_id": task_id,
                "dataset_task_index": dataset_task_index,
                "task": task.language,
                "episode": episode,
                "success": success,
                "steps": steps,
                "video": str(video_path),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=True))

    metrics = {
        "checkpoint": args.checkpoint,
        "suite": args.suite,
        "task_ids": [task_id for task_id, _ in task_specs],
        "dataset_task_indices": [dataset_task_index for _, dataset_task_index in task_specs],
        "episodes": len(results),
        "successes": sum(int(result["success"]) for result in results),
        "success_rate": sum(int(result["success"]) for result in results) / max(len(results), 1),
        "results": results,
    }
    metrics_path = video_dir / "rollout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "results"}, indent=2))
    print(f"saved_metrics={metrics_path}")


def resolve_task_specs(
    suite: Any,
    task_ids: list[int] | None,
    dataset_task_indices: list[int] | None,
    repo_id: str,
    revision: str | None,
) -> list[tuple[int, int | None]]:
    if task_ids is not None:
        return [(task_id, None) for task_id in task_ids]
    assert dataset_task_indices is not None
    return map_dataset_tasks_to_suite(suite, dataset_task_indices, read_dataset_tasks(repo_id, revision))


def read_dataset_tasks(repo_id: str, revision: str | None) -> list[tuple[int, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to map dataset task indices to LIBERO suite-local ids") from exc

    path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/tasks.parquet",
        repo_type="dataset",
        revision=revision,
    )
    rows = pq.read_table(path).to_pylist()
    return [(int(row["task_index"]), str(row["__index_level_0__"])) for row in rows]


def map_dataset_tasks_to_suite(
    suite: Any,
    dataset_task_indices: list[int],
    dataset_tasks: list[tuple[int, str]],
) -> list[tuple[int, int]]:
    descriptions = dict(dataset_tasks)
    suite_task_ids = {_normalize_instruction(task.language): task_id for task_id, task in enumerate(suite.tasks)}
    task_specs = []
    for dataset_task_index in dataset_task_indices:
        if dataset_task_index not in descriptions:
            raise ValueError(f"Unknown dataset task index: {dataset_task_index}")
        instruction = descriptions[dataset_task_index]
        task_id = suite_task_ids.get(_normalize_instruction(instruction))
        if task_id is None:
            raise ValueError(
                f"Dataset task {dataset_task_index} ({instruction!r}) is not part of suite {type(suite).__name__}"
            )
        task_specs.append((task_id, dataset_task_index))
    return task_specs


def _normalize_instruction(instruction: str) -> str:
    return " ".join(instruction.lower().split())


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_steps: int | None,
) -> tuple[RepresentationEncoder, PI0LiteFlowPolicy]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    clip_model = train_args.get("clip_model", "openai/clip-vit-base-patch32")
    image_encoder = CLIPImageEncoder(clip_model, freeze=True)
    text_encoder = CLIPTextEncoder(clip_model, freeze=True)
    encoder = RepresentationEncoder(
        RepresentationEncoderConfig(
            image_dim=image_encoder.output_dim,
            text_dim=text_encoder.output_dim,
            state_dim=int(train_args.get("state_dim", 8)),
            cond_dim=int(train_args.get("cond_dim", 256)),
            num_views=2,
        ),
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    ).to(device)
    _, unexpected = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    unexpected = [key for key in unexpected if not key.startswith(("image_encoder.model.", "text_encoder.model."))]
    if unexpected:
        raise RuntimeError(f"Unexpected encoder checkpoint keys: {unexpected}")

    policy = PI0LiteFlowPolicy(
        PI0LiteFlowConfig(
            horizon=int(train_args.get("horizon", 16)),
            action_dim=int(train_args.get("action_dim", 7)),
            cond_dim=int(train_args.get("cond_dim", 256)),
            inference_steps=int(num_steps or train_args.get("inference_steps", 8)),
        )
    ).to(device)
    policy.load_state_dict(checkpoint["policy"])
    encoder.eval()
    policy.eval()
    return encoder, policy


def load_init_states(suite: Any, task_id: int, init_states_root: str) -> Any:
    task = suite.tasks[task_id]
    init_states_file = Path(task.init_states_file)
    path = Path(init_states_root) / task.problem_folder / init_states_file.name
    return torch.load(path, weights_only=False)


def reset_env(env: Any, init_state: Any, seed: int, warmup_steps: int) -> dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(warmup_steps):
        obs, _, _, _ = env.step(NO_OP_ACTION)
    for robot in env.robots:
        robot.controller.use_delta = True
    return obs


def observation_to_encoder_batch(obs: dict[str, Any], instruction: str, device: torch.device) -> dict[str, Any]:
    agentview = rotate_image(obs["agentview_image"])
    wrist = rotate_image(obs["robot0_eye_in_hand_image"])
    images = np.stack([agentview, wrist], axis=0)
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"]),
            quaternion_to_axis_angle(np.asarray(obs["robot0_eef_quat"])),
            np.asarray(obs["robot0_gripper_qpos"]),
        ]
    )
    return {
        "images": torch.from_numpy(images).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0,
        "state": torch.from_numpy(state).float().unsqueeze(0).to(device),
        "instruction": [instruction],
    }


def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return quat[:3] / den * (2.0 * np.arccos(w))


def rotate_image(image: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def make_video_frame(obs: dict[str, Any], view: str) -> np.ndarray:
    agentview = rotate_image(obs["agentview_image"])
    if view == "agentview":
        return agentview
    wrist = rotate_image(obs["robot0_eye_in_hand_image"])
    if view == "wrist":
        return wrist
    return np.concatenate([agentview, wrist], axis=1)


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write an empty video")
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to write {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

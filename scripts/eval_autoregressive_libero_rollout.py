#!/usr/bin/env python3
"""Headless LIBERO success-rate rollout for the CLIP autoregressive policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from eval_libero_rollout import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    load_init_states,
    make_video_frame,
    observation_to_encoder_batch,
    prepare_libero_runtime_paths,
    reset_env,
    resolve_task_specs,
    set_seed,
    summarize_action_trace,
    write_mp4,
)
from lerobot_policy_pi0_lite_flow.autoregressive_experiment import (  # noqa: E402
    COND_KEY,
    build_encoder,
    build_policy,
    load_trainable_encoder_state_dict,
    resolve_device,
    tensor_dict_from_lists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--suite", default="libero_object")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-ids", nargs="+", type=int, help="Task ids local to the selected suite.")
    task_group.add_argument(
        "--dataset-task-indices",
        nargs="+",
        type=int,
        help="Global Hugging Face dataset task indices, mapped to suite-local ids by instruction text.",
    )
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-view", choices=("agentview", "wrist", "both"), default="agentview")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    prepare_libero_runtime_paths()
    set_seed(args.seed)
    device = resolve_device(args.device)
    encoder, policy = load_model(args.checkpoint, device)

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
    video_dir = args.video_dir or args.checkpoint.parent / "rollout_videos"
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
                action_trace = []
                success = False
                steps = 0
                for steps in range(1, max_steps + 1):
                    frames.append(make_video_frame(obs, args.video_view))
                    batch = observation_to_encoder_batch(obs, task.language, device)
                    with torch.no_grad():
                        cond = encoder(batch)
                        action = policy.select_action({COND_KEY: cond}, temperature=args.temperature)
                    action_numpy = np.clip(action[0].detach().cpu().numpy(), -1.0, 1.0)
                    action_trace.append(action_numpy)
                    obs, _, done, _ = env.step(action_numpy)
                    success = bool(done or env.check_success())
                    if success:
                        frames.append(make_video_frame(obs, args.video_view))
                        break
            finally:
                env.close()

            outcome = "success" if success else "failure"
            video_path = video_dir / f"{args.suite}_task{task_id:02d}_episode{episode:02d}_{outcome}.mp4"
            write_mp4(video_path, frames, args.fps)
            result = {
                "suite": args.suite,
                "task_id": task_id,
                "dataset_task_index": dataset_task_index,
                "task": task.language,
                "episode": episode,
                "success": success,
                "steps": steps,
                **summarize_action_trace(action_trace),
                "video": str(video_path),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=True))

    metrics = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "suite": args.suite,
        "task_ids": [task_id for task_id, _ in task_specs],
        "dataset_task_indices": [dataset_task_index for _, dataset_task_index in task_specs],
        "episodes": len(results),
        "successes": sum(int(result["success"]) for result in results),
        "success_rate": sum(int(result["success"]) for result in results) / max(len(results), 1),
        "results": results,
    }
    metrics_path = video_dir / "rollout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "results"}, indent=2))
    print(f"saved_metrics={metrics_path}")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    action_stats = tensor_dict_from_lists(checkpoint["action_stats"])
    encoder = build_encoder(train_args).to(device)
    policy = build_policy(train_args, action_stats).to(device)
    load_trainable_encoder_state_dict(encoder, checkpoint["encoder_state_dict"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    encoder.eval()
    policy.eval()
    return encoder, policy


if __name__ == "__main__":
    main()

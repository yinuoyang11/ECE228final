"""Simulation rollout evaluation for BC Regression on LIBERO.

Loads a trained BCRegressionPolicy checkpoint, runs N episodes per task inside
the LIBERO MuJoCo environment, and reports per-episode success/failure as well
as an aggregate success_rate.

Designed to run on a Linux server with:
  - LIBERO installed from source (pip install -e /path/to/LIBERO)
  - robosuite == 1.4.0
  - MuJoCo with EGL headless rendering (MUJOCO_GL=egl)
  - ffmpeg on PATH (for MP4 recording)

Usage example (server):
  python scripts/sim_rollout_regression.py \\
    results/task202122_regression/checkpoint_final.pt \\
    --dataset-task-indices 20 21 22 \\
    --episodes-per-task 10 \\
    --video-dir results/sim_rollout \\
    --device cuda

Per-episode results and success_rate are printed as JSON and saved to
  <video-dir>/rollout_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.configuration_regression import BCRegressionConfig
from lerobot_policy_pi0_lite_flow.modeling_regression import COND_KEY, BCRegressionPolicy
from lerobot_policy_pi0_lite_flow.representation_encoder import (
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)

DEFAULT_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal":   300,
    "libero_10":     520,
    "libero_90":     400,
}
NO_OP_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RepresentationEncoder, BCRegressionPolicy]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ta = checkpoint.get("args", {})
    clip_model = ta.get("clip_model", "openai/clip-vit-base-patch32")

    ie = CLIPImageEncoder(clip_model, freeze=True)
    te = CLIPTextEncoder(clip_model, freeze=True)
    encoder = RepresentationEncoder(
        RepresentationEncoderConfig(
            image_dim=ie.output_dim,
            text_dim=te.output_dim,
            state_dim=int(ta.get("state_dim", 8)),
            cond_dim=int(ta.get("cond_dim", 256)),
            num_views=2,
        ),
        image_encoder=ie,
        text_encoder=te,
    ).to(device)
    _, unexpected = encoder.load_state_dict(checkpoint["encoder"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith(("image_encoder.model.", "text_encoder.model."))]
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")

    policy = BCRegressionPolicy(
        BCRegressionConfig(
            horizon=int(ta.get("horizon", 16)),
            action_dim=int(ta.get("action_dim", 7)),
            cond_dim=int(ta.get("cond_dim", 256)),
            hidden_dim=int(ta.get("hidden_dim", 256)),
            num_layers=int(ta.get("num_layers", 4)),
            dropout=float(ta.get("dropout", 0.0)),
        )
    ).to(device)
    policy.load_state_dict(checkpoint["policy"])
    encoder.eval()
    policy.eval()
    return encoder, policy


# ── observation helpers ───────────────────────────────────────────────────────
def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return quat[:3] / den * (2.0 * np.arccos(w))


def rotate_image(image: Any) -> np.ndarray:
    """LIBERO images are stored upside-down; flip back."""
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def obs_to_encoder_batch(obs: dict[str, Any], instruction: str, device: torch.device) -> dict:
    agentview = rotate_image(obs["agentview_image"])         # (H,W,3) uint8
    wrist     = rotate_image(obs["robot0_eye_in_hand_image"])
    images = np.stack([agentview, wrist], axis=0)            # (2,H,W,3)
    state = np.concatenate([
        np.asarray(obs["robot0_eef_pos"]),
        quaternion_to_axis_angle(np.asarray(obs["robot0_eef_quat"])),
        np.asarray(obs["robot0_gripper_qpos"]),
    ]).astype(np.float32)                                    # (8,)
    return {
        "images":      torch.from_numpy(images).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0,
        "state":       torch.from_numpy(state).unsqueeze(0).to(device),
        "instruction": [instruction],
    }


def make_video_frame(obs: dict[str, Any], view: str) -> np.ndarray:
    agentview = rotate_image(obs["agentview_image"])
    if view == "agentview":
        return agentview
    wrist = rotate_image(obs["robot0_eye_in_hand_image"])
    if view == "wrist":
        return wrist
    return np.concatenate([agentview, wrist], axis=1)


# ── video writing ─────────────────────────────────────────────────────────────
def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}", "-framerate", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


# ── environment helpers ───────────────────────────────────────────────────────
def load_init_states(suite: Any, task_id: int, init_states_root: str) -> Any:
    task = suite.tasks[task_id]
    path = Path(init_states_root) / task.problem_folder / Path(task.init_states_file).name
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


# ── task resolution helpers ───────────────────────────────────────────────────
def read_dataset_tasks(repo_id: str, revision: str | None) -> list[tuple[int, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("pyarrow required: pip install pyarrow") from e
    path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/tasks.parquet",
        repo_type="dataset",
        revision=revision,
    )
    rows = pq.read_table(path).to_pylist()
    return [(int(r["task_index"]), str(r["__index_level_0__"])) for r in rows]


def map_dataset_tasks_to_suite(
    suite: Any,
    dataset_task_indices: list[int],
    dataset_tasks: list[tuple[int, str]],
) -> list[tuple[int, int]]:
    descriptions = dict(dataset_tasks)
    suite_map = {
        " ".join(task.language.lower().split()): tid
        for tid, task in enumerate(suite.tasks)
    }
    specs = []
    for dti in dataset_task_indices:
        if dti not in descriptions:
            raise ValueError(f"Unknown dataset task index: {dti}")
        instr = " ".join(descriptions[dti].lower().split())
        tid = suite_map.get(instr)
        if tid is None:
            raise ValueError(
                f"Dataset task {dti} ({descriptions[dti]!r}) not found in suite {type(suite).__name__}"
            )
        specs.append((tid, dti))
    return specs


# ── action trace summary ──────────────────────────────────────────────────────
def summarize_action_trace(trace: list[np.ndarray]) -> dict[str, float]:
    arr = np.asarray(trace)   # (T, 7)
    g = arr[:, 6]
    return {
        "gripper_min":           float(g.min()),
        "gripper_max":           float(g.max()),
        "gripper_mean":          float(g.mean()),
        "gripper_close_fraction": float((g >= 0).mean()),
    }


# ── seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation rollout evaluation for BCRegressionPolicy on LIBERO. "
                    "Requires Linux + MuJoCo EGL + ffmpeg."
    )
    parser.add_argument("checkpoint", help="Path to checkpoint_final.pt")
    parser.add_argument("--suite", default="libero_object",
                        help="LIBERO benchmark suite (default: libero_object)")

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-ids", nargs="+", type=int,
                            help="Suite-local task ids (0-indexed within the suite)")
    task_group.add_argument("--dataset-task-indices", nargs="+", type=int,
                            help="Global HuggingFace dataset task indices (e.g. 20 21 22)")

    parser.add_argument("--repo-id",          default="HuggingFaceVLA/libero")
    parser.add_argument("--revision",         default=None)
    parser.add_argument("--episodes-per-task", type=int, default=10,
                        help="How many episodes to run per task (default: 10)")
    parser.add_argument("--max-steps",        type=int, default=None,
                        help="Max env steps per episode. Defaults per suite.")
    parser.add_argument("--image-size",       type=int, default=256)
    parser.add_argument("--fps",              type=int, default=10)
    parser.add_argument("--warmup-steps",     type=int, default=10,
                        help="No-op steps after env reset to let physics settle")
    parser.add_argument("--video-dir",        default=None,
                        help="Where to save MP4s and rollout_metrics.json. "
                             "Default: <checkpoint_dir>/sim_rollout")
    parser.add_argument("--video-view",       choices=("agentview", "wrist", "both"),
                        default="both")
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",             type=int, default=7)
    args = parser.parse_args()

    # ── env setup ──────────────────────────────────────────────────────────────
    os.environ.setdefault("MUJOCO_GL",        "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    encoder, policy = load_model(Path(args.checkpoint), device)
    print("Model loaded.")

    # ── LIBERO imports (server only) ───────────────────────────────────────────
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        raise SystemExit(
            "LIBERO not found. On the server run:\n"
            "  pip install -e /path/to/LIBERO\n"
            f"Original error: {e}"
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        raise ValueError(
            f"Unknown suite {args.suite!r}. "
            f"Available: {sorted(benchmark_dict)}"
        )
    suite = benchmark_dict[args.suite]()

    if args.task_ids is not None:
        task_specs = [(tid, None) for tid in args.task_ids]
    else:
        task_specs = map_dataset_tasks_to_suite(
            suite,
            args.dataset_task_indices,
            read_dataset_tasks(args.repo_id, args.revision),
        )

    max_steps  = args.max_steps or DEFAULT_MAX_STEPS.get(args.suite, 500)
    video_dir  = Path(args.video_dir) if args.video_dir else Path(args.checkpoint).parent / "sim_rollout"
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Suite:          {args.suite}")
    print(f"Tasks:          {task_specs}")
    print(f"Episodes/task:  {args.episodes_per_task}")
    print(f"Max steps/ep:   {max_steps}")
    print(f"Video dir:      {video_dir}")
    print("=" * 60)

    results = []
    for task_id, dataset_task_index in task_specs:
        task        = suite.get_task(task_id)
        init_states = load_init_states(suite, task_id, get_libero_path("init_states"))
        bddl_file   = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        print(f"\nTask {task_id} ({dataset_task_index}): {task.language}")

        for episode in range(args.episodes_per_task):
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                camera_heights=args.image_size,
                camera_widths=args.image_size,
            )
            try:
                init_state = init_states[episode % len(init_states)]
                obs        = reset_env(env, init_state, args.seed + episode, args.warmup_steps)
                frames: list[np.ndarray] = []
                action_trace: list[np.ndarray] = []
                success = False
                steps   = 0

                for steps in range(1, max_steps + 1):
                    frames.append(make_video_frame(obs, args.video_view))

                    batch = obs_to_encoder_batch(obs, task.language, device)
                    with torch.no_grad():
                        cond   = encoder(batch)
                        action = policy.select_action({COND_KEY: cond})  # (1, 7)

                    action_numpy = np.clip(action[0].detach().cpu().numpy(), -1.0, 1.0)
                    action_trace.append(action_numpy)
                    obs, _, done, _ = env.step(action_numpy)
                    success = bool(done or env.check_success())
                    if success:
                        frames.append(make_video_frame(obs, args.video_view))
                        break

            finally:
                env.close()

            outcome    = "success" if success else "failure"
            video_path = video_dir / f"{args.suite}_task{task_id:02d}_ep{episode:02d}_{outcome}.mp4"
            write_mp4(video_path, frames, args.fps)

            result = {
                "suite":               args.suite,
                "task_id":             task_id,
                "dataset_task_index":  dataset_task_index,
                "task":                task.language,
                "episode":             episode,
                "success":             success,
                "steps":               steps,
                **summarize_action_trace(action_trace),
                "video":               str(video_path),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=True))

    # ── aggregate ──────────────────────────────────────────────────────────────
    successes    = sum(int(r["success"]) for r in results)
    success_rate = successes / max(len(results), 1)

    metrics = {
        "checkpoint":           str(args.checkpoint),
        "suite":                args.suite,
        "task_ids":             [tid for tid, _ in task_specs],
        "dataset_task_indices": [dti for _, dti in task_specs],
        "episodes":             len(results),
        "successes":            successes,
        "success_rate":         success_rate,
        "results":              results,
    }
    metrics_path = video_dir / "rollout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = {k: v for k, v in metrics.items() if k != "results"}
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nVideos + metrics saved to: {video_dir}")


if __name__ == "__main__":
    main()

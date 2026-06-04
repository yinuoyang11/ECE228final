"""Offline rollout visualization for BC Regression on LIBERO Task 20.

Loads a trained checkpoint, runs inference on real dataset frames, and
generates an MP4 video showing:
  - Left: agentview camera frame
  - Right: wrist camera frame
  - Bottom bar: predicted vs actual action for each of the 7 DOF
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.configuration_regression import BCRegressionConfig
from lerobot_policy_pi0_lite_flow.libero_adapter import LIBEROActionChunkDataset
from lerobot_policy_pi0_lite_flow.modeling_regression import ACTION, COND_KEY, BCRegressionPolicy
from lerobot_policy_pi0_lite_flow.representation_encoder import (
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)

ACTION_LABELS = ["X", "Y", "Z", "Rx", "Ry", "Rz", "Grip"]
COLOR_PRED   = (52, 152, 219)   # blue  BGR
COLOR_ACTUAL = (46, 204, 113)   # green BGR
FONT         = cv2.FONT_HERSHEY_SIMPLEX


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ta   = ckpt.get("args", {})
    clip = ta.get("clip_model", "openai/clip-vit-base-patch32")

    ie = CLIPImageEncoder(clip, freeze=True)
    te = CLIPTextEncoder(clip, freeze=True)
    encoder = RepresentationEncoder(
        RepresentationEncoderConfig(
            image_dim=ie.output_dim, text_dim=te.output_dim,
            state_dim=int(ta.get("state_dim", 8)),
            cond_dim=int(ta.get("cond_dim", 256)),
            num_views=2,
        ),
        image_encoder=ie, text_encoder=te,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=False)

    policy = BCRegressionPolicy(
        BCRegressionConfig(
            horizon=int(ta.get("horizon", 16)),
            action_dim=int(ta.get("action_dim", 7)),
            cond_dim=int(ta.get("cond_dim", 256)),
            hidden_dim=int(ta.get("hidden_dim", 256)),
            num_layers=int(ta.get("num_layers", 4)),
        )
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    encoder.eval(); policy.eval()
    return encoder, policy


def tensor_to_bgr(img_chw: torch.Tensor) -> np.ndarray:
    """(3, H, W) float[0,1] → (H, W, 3) uint8 BGR"""
    arr = (img_chw.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_action_bar(pred: np.ndarray, actual: np.ndarray, width: int, bar_h: int = 120) -> np.ndarray:
    """Draw a side-by-side bar chart comparing pred vs actual first action step."""
    canvas = np.full((bar_h, width, 3), 30, dtype=np.uint8)
    n = len(ACTION_LABELS)
    col_w = width // n
    max_val = 1.0

    for i, label in enumerate(ACTION_LABELS):
        x0 = i * col_w
        cx = x0 + col_w // 2

        # actual bar (green, left of center)
        a_val = float(actual[i])
        a_h   = int(abs(a_val) / max_val * (bar_h // 2 - 18))
        a_top = bar_h // 2 - a_h if a_val >= 0 else bar_h // 2
        cv2.rectangle(canvas, (cx - col_w // 2 + 4, a_top),
                      (cx - 4, a_top + a_h), COLOR_ACTUAL, -1)

        # predicted bar (blue, right of center, overlaid)
        p_val = float(pred[i])
        p_h   = int(abs(p_val) / max_val * (bar_h // 2 - 18))
        p_top = bar_h // 2 - p_h if p_val >= 0 else bar_h // 2
        cv2.rectangle(canvas, (cx - 4, p_top),
                      (cx + col_w // 2 - 4, p_top + p_h), COLOR_PRED, -1)

        # center line
        cv2.line(canvas, (x0, bar_h // 2), (x0 + col_w, bar_h // 2), (80, 80, 80), 1)
        # label
        cv2.putText(canvas, label, (cx - 8, bar_h - 6), FONT, 0.42, (200, 200, 200), 1)

    # legend
    cv2.rectangle(canvas, (4, 4), (16, 14), COLOR_ACTUAL, -1)
    cv2.putText(canvas, "Actual", (20, 14), FONT, 0.38, (200, 200, 200), 1)
    cv2.rectangle(canvas, (74, 4), (86, 14), COLOR_PRED, -1)
    cv2.putText(canvas, "Predicted", (90, 14), FONT, 0.38, (200, 200, 200), 1)

    return canvas


def make_frame(agent_bgr, wrist_bgr, pred_act, actual_act, step, instruction, frame_h=256):
    """Compose one video frame: [agent | wrist] + action bar"""
    frame_w = frame_h * 2
    top  = np.concatenate([agent_bgr, wrist_bgr], axis=1)   # (H, 2W, 3)
    bar  = draw_action_bar(pred_act, actual_act, frame_w)

    # overlay info text
    cv2.putText(top, f"Step {step:3d}", (8, 22), FONT, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(top, instruction[:55], (8, 44), FONT, 0.42, (200,255,200), 1, cv2.LINE_AA)
    cv2.putText(top, "Agent view", (8, frame_h - 8), FONT, 0.38, (160,160,160), 1)
    cv2.putText(top, "Wrist view", (frame_h + 8, frame_h - 8), FONT, 0.38, (160,160,160), 1)

    return np.concatenate([top, bar], axis=0)  # (H+bar, 2W, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="results/task20_regression/checkpoint_final.pt")
    parser.add_argument("--local-dir",   default="D:/AA_Graduate/datasets/libero")
    parser.add_argument("--task-indices", nargs="+", type=int, default=[20])
    parser.add_argument("--num-frames",  type=int, default=300, help="Max frames to render")
    parser.add_argument("--fps",         type=int, default=10)
    parser.add_argument("--output",      default="results/task20_rollout.mp4")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = ROOT / args.checkpoint
    out_path  = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {ckpt_path} ...")
    encoder, policy = load_model(ckpt_path, device)

    print(f"Loading dataset (task {args.task_indices}) ...")
    dataset = LIBEROActionChunkDataset(
        repo_id="HuggingFaceVLA/libero",
        task_indices=args.task_indices,
        root=args.local_dir,
        horizon=1,           # we only need the current-step action
    )

    total_frames = min(args.num_frames, len(dataset))
    print(f"Rendering {total_frames} frames → {out_path}")

    frame_h = 256
    frame_w = frame_h * 2
    bar_h   = 120
    writer  = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (frame_w, frame_h + bar_h),
    )

    results = []
    for idx in range(total_frames):
        sample = dataset[idx]
        images      = sample["images"].unsqueeze(0).to(device)        # (1,2,3,H,W)
        state       = sample["state"].unsqueeze(0).to(device)         # (1,8)
        instruction = sample["instruction"]
        actual_act  = sample["action"][0].cpu().numpy()               # (7,)

        batch = {"images": images, "state": state, "instruction": [instruction]}
        with torch.no_grad():
            cond = encoder(batch)
            pred_act = policy.select_action({COND_KEY: cond})         # (1,7)
        pred_act = pred_act[0].cpu().numpy()

        agent_bgr = tensor_to_bgr(sample["images"][0])  # view 0
        wrist_bgr = tensor_to_bgr(sample["images"][1])  # view 1

        agent_bgr = cv2.resize(agent_bgr, (frame_h, frame_h))
        wrist_bgr = cv2.resize(wrist_bgr, (frame_h, frame_h))

        frame = make_frame(agent_bgr, wrist_bgr, pred_act, actual_act,
                           idx, instruction, frame_h)
        writer.write(frame)

        mse = float(((pred_act - actual_act) ** 2).mean())
        results.append({"step": idx, "mse": mse})
        if idx % 50 == 0:
            print(f"  frame {idx}/{total_frames}  step_mse={mse:.4f}")

    writer.release()
    avg_mse = float(np.mean([r["mse"] for r in results]))
    print(f"\nDone. avg_mse={avg_mse:.5f}")
    print(f"Video saved: {out_path}")

    metrics_path = out_path.parent / "rollout_metrics.json"
    metrics_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "task_indices": args.task_indices,
        "num_frames": total_frames,
        "avg_mse": avg_mse,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()

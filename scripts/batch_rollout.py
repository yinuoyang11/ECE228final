"""Generate 10 offline rollout videos covering different task scenarios.

Tasks 20, 21, 22 × varied episodes → 10 MP4 files saved in results/rollouts/.
Each video shows: agent-view | wrist-view + per-DOF action bar (predicted vs actual).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot_policy_pi0_lite_flow.configuration_regression import BCRegressionConfig
from lerobot_policy_pi0_lite_flow.libero_adapter import (
    LIBEROActionChunkDataset,
    resolve_task_episodes,
)
from lerobot_policy_pi0_lite_flow.modeling_regression import ACTION, COND_KEY, BCRegressionPolicy
from lerobot_policy_pi0_lite_flow.representation_encoder import (
    CLIPImageEncoder,
    CLIPTextEncoder,
    RepresentationEncoder,
    RepresentationEncoderConfig,
)

# ── visual constants ─────────────────────────────────────────────────────────
ACTION_LABELS = ["X", "Y", "Z", "Rx", "Ry", "Rz", "Grip"]
C_PRED   = (52,  152, 219)   # blue  BGR
C_ACTUAL = (39,  174,  96)   # green BGR
C_BG     = (245, 247, 250)   # near-white BGR
C_ZERO   = (180, 180, 180)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

FRAME_H  = 256
FRAME_W  = FRAME_H * 2
BAR_H    = 110
OUT_H    = FRAME_H + BAR_H

# 10 scenarios: (task_idx, episode_offset_frac)
# fraction from 0→1 picks which portion of that task's samples to use
SCENARIOS = [
    # task 20 – "pick up orange juice"
    dict(task=20, frac=0.00, label="Task20-ep0"),
    dict(task=20, frac=0.33, label="Task20-ep15"),
    dict(task=20, frac=0.66, label="Task20-ep30"),
    # task 21 – "pick up ketchup"
    dict(task=21, frac=0.00, label="Task21-ep0"),
    dict(task=21, frac=0.25, label="Task21-ep11"),
    dict(task=21, frac=0.50, label="Task21-ep22"),
    dict(task=21, frac=0.75, label="Task21-ep34"),
    # task 22 – "pick up cream cheese"
    dict(task=22, frac=0.00, label="Task22-ep0"),
    dict(task=22, frac=0.40, label="Task22-ep18"),
    dict(task=22, frac=0.75, label="Task22-ep33"),
]
assert len(SCENARIOS) == 10


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ta = ckpt.get("args", {})
    clip = ta.get("clip_model", "openai/clip-vit-base-patch32")

    ie = CLIPImageEncoder(clip, freeze=True)
    te = CLIPTextEncoder(clip, freeze=True)
    enc = RepresentationEncoder(
        RepresentationEncoderConfig(
            image_dim=ie.output_dim, text_dim=te.output_dim,
            state_dim=int(ta.get("state_dim", 8)),
            cond_dim=int(ta.get("cond_dim", 256)),
            num_views=2,
        ),
        image_encoder=ie, text_encoder=te,
    ).to(device)
    enc.load_state_dict(ckpt["encoder"], strict=False)

    pol = BCRegressionPolicy(
        BCRegressionConfig(
            horizon=int(ta.get("horizon", 16)),
            action_dim=int(ta.get("action_dim", 7)),
            cond_dim=int(ta.get("cond_dim", 256)),
            hidden_dim=int(ta.get("hidden_dim", 256)),
            num_layers=int(ta.get("num_layers", 4)),
        )
    ).to(device)
    pol.load_state_dict(ckpt["policy"])
    enc.eval(); pol.eval()
    return enc, pol


# ── frame helpers ─────────────────────────────────────────────────────────────
def tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_bar(pred: np.ndarray, actual: np.ndarray, w: int, h: int = BAR_H) -> np.ndarray:
    canvas = np.full((h, w, 3), 250, dtype=np.uint8)
    n = len(ACTION_LABELS)
    col_w = w // n
    mid = h // 2 - 12   # vertical center of bar region
    scale = (mid - 18) / 1.0

    for i, lbl in enumerate(ACTION_LABELS):
        x0 = i * col_w
        cx = x0 + col_w // 2

        def bar_rect(val, color, left):
            vh = int(abs(val) * scale)
            if val >= 0:
                y1 = mid - vh; y2 = mid
            else:
                y1 = mid; y2 = mid + vh
            if left:
                cv2.rectangle(canvas, (x0+2, y1), (cx-2, y2), color, -1)
            else:
                cv2.rectangle(canvas, (cx+2, y1), (x0+col_w-2, y2), color, -1)

        bar_rect(float(actual[i]), C_ACTUAL, left=True)
        bar_rect(float(pred[i]),   C_PRED,   left=False)

        # zero line
        cv2.line(canvas, (x0, mid), (x0+col_w, mid), C_ZERO, 1)
        # label
        cv2.putText(canvas, lbl, (cx-7, h-5), FONT, 0.38, (80,80,80), 1, cv2.LINE_AA)
        # separator
        cv2.line(canvas, (x0, 0), (x0, h), (220,220,220), 1)

    # legend
    cv2.rectangle(canvas, (4,  3), (14, 10), C_ACTUAL, -1)
    cv2.putText(canvas, "Expert",    (17,  10), FONT, 0.35, (60,60,60), 1)
    cv2.rectangle(canvas, (70, 3), (80, 10), C_PRED,   -1)
    cv2.putText(canvas, "Predicted", (83,  10), FONT, 0.35, (60,60,60), 1)
    return canvas


def make_frame(agent_bgr, wrist_bgr, pred, actual, step, instruction, task_name):
    top = np.concatenate([
        cv2.resize(agent_bgr, (FRAME_H, FRAME_H)),
        cv2.resize(wrist_bgr, (FRAME_H, FRAME_H)),
    ], axis=1)
    bar = draw_bar(pred, actual, FRAME_W)

    # task/step text
    cv2.putText(top, f"[{task_name}]  Step {step:4d}",
                (8, 22), FONT, 0.52, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(top, instruction[:62],
                (8, 42), FONT, 0.40, (180,255,180), 1, cv2.LINE_AA)
    cv2.putText(top, "Agent view",
                (8, FRAME_H-8), FONT, 0.36, (160,160,160), 1)
    cv2.putText(top, "Wrist view",
                (FRAME_H+8, FRAME_H-8), FONT, 0.36, (160,160,160), 1)
    return np.concatenate([top, bar], axis=0)


# ── per-scenario renderer ──────────────────────────────────────────────────────
def render_scenario(sc: dict, enc, pol, device, local_dir, frames_per_video, fps, out_dir):
    task_idx = sc["task"]
    frac = sc["frac"]
    label = sc["label"]
    out_path = out_dir / f"{label}.mp4"

    ds = LIBEROActionChunkDataset(
        repo_id="HuggingFaceVLA/libero",
        task_indices=[task_idx],
        root=local_dir,
        horizon=1,
    )
    n = len(ds)
    start = int(n * frac)
    end   = min(start + frames_per_video, n)

    instruction = ds[start]["instruction"]
    task_names  = {20: "OJ-Basket", 21: "Ketchup-Basket", 22: "CrmChz-Basket"}
    task_name   = task_names.get(task_idx, f"Task{task_idx}")

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (FRAME_W, OUT_H),
    )
    mse_list = []
    for idx in range(start, end):
        sample = ds[idx]
        images = sample["images"].unsqueeze(0).to(device)
        state  = sample["state"].unsqueeze(0).to(device)
        actual = sample["action"][0].cpu().numpy()

        with torch.no_grad():
            cond = enc({"images": images, "state": state, "instruction": [instruction]})
            pred = pol.select_action({COND_KEY: cond})[0].cpu().numpy()

        frame = make_frame(
            tensor_to_bgr(sample["images"][0]),
            tensor_to_bgr(sample["images"][1]),
            pred, actual, idx - start, instruction, task_name,
        )
        writer.write(frame)
        mse_list.append(float(((pred - actual) ** 2).mean()))

    writer.release()
    avg_mse = float(np.mean(mse_list)) if mse_list else 0.0
    print(f"  [{label}] frames={end-start}  avg_mse={avg_mse:.5f}  → {out_path.name}")
    return {"scenario": label, "task": task_idx, "start": start, "frames": end - start, "avg_mse": avg_mse}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/task202122_regression/checkpoint_final.pt")
    parser.add_argument("--local-dir",  default="D:/AA_Graduate/datasets/libero")
    parser.add_argument("--frames",     type=int, default=200, help="Frames per video")
    parser.add_argument("--fps",        type=int, default=10)
    parser.add_argument("--out-dir",    default="results/rollouts")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt   = ROOT / args.checkpoint
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {ckpt}")
    enc, pol = load_model(ckpt, device)

    all_stats = []
    for i, sc in enumerate(SCENARIOS):
        print(f"\n[{i+1}/10] Rendering scenario: {sc['label']} ...")
        stat = render_scenario(sc, enc, pol, device, args.local_dir, args.frames, args.fps, out_dir)
        all_stats.append(stat)

    summary_path = out_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    avg = float(np.mean([s["avg_mse"] for s in all_stats]))
    print(f"\n{'='*55}")
    print(f"All 10 videos saved to: {out_dir}")
    print(f"Overall avg_mse across all videos: {avg:.5f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

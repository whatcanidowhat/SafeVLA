"""B1 shadow logger: score baseline actions without changing them."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from online_evaluation.shadow_safety_predictor import (
    HeuristicSafetyPredictor,
    R1PredictiveRewardSystem,
    extract_depth_mean,
    extract_metadata,
)


def shadow_predictor_enabled() -> bool:
    return os.getenv("SHADOW_PREDICTOR_ENABLE", "0") == "1"


def default_shadow_logdir(outdir: Optional[str] = None) -> str:
    env = os.getenv("SHADOW_PREDICTOR_LOGDIR", "").strip()
    if env:
        return env
    if outdir:
        return os.path.join(outdir, "shadow_logs")
    return "./eval/shadow_logs"


class ShadowPredictorLogger:
    """Per-worker JSONL logger for heuristic safety predictions."""

    def __init__(
        self,
        outdir: str,
        worker_id: int,
        enabled: Optional[bool] = None,
    ):
        self.enabled = shadow_predictor_enabled() if enabled is None else bool(enabled)
        self.outdir = Path(outdir)
        self.worker_id = worker_id
        self.score_all_actions = os.getenv("SHADOW_SCORE_ALL_ACTIONS", "0") == "1"

        self.predictor = HeuristicSafetyPredictor()
        self.reward_system = R1PredictiveRewardSystem()
        self.obs_history: List[Dict[str, Any]] = []
        self.last_action = "None"
        self.records: List[Dict[str, Any]] = []
        self._episode_meta: Dict[str, Any] = {}
        self._depth_warning_printed = False

        if self.enabled:
            self.outdir.mkdir(parents=True, exist_ok=True)
            print(
                f"[Shadow] Logger enabled worker={worker_id} logdir={self.outdir}",
                flush=True,
            )

    def reset_episode(self, episode_meta: Optional[Dict[str, Any]] = None):
        self.predictor = HeuristicSafetyPredictor()
        self.reward_system = R1PredictiveRewardSystem()
        self.obs_history = []
        self.last_action = "None"
        self.records = []
        self._episode_meta = dict(episode_meta or {})

    def shadow_score(
        self,
        observations: Dict[str, Any],
        goal: str,
        action_list: Sequence[str],
        baseline_action: str,
    ) -> Dict[str, Any]:
        """Read-only score for the baseline action. Caller must pass a shallow copy."""
        meta = extract_metadata(observations)
        if "depth_mean" not in meta and not self._depth_warning_printed:
            print(
                "[Shadow][Warning] No depth in shadow obs; depth_mean defaults to 10.0.",
                flush=True,
            )
            self._depth_warning_printed = True

        risks, progress, msg = self.predictor.predict_heuristic(
            self.obs_history, baseline_action, current_obs_metadata=meta
        )
        reward = self.reward_system.calculate_predicted_reward(
            risks, progress, baseline_action
        )

        all_action_scores = None
        if self.score_all_actions and action_list:
            all_action_scores = []
            for a in action_list:
                rsk, prog, _ = self.predictor.predict_heuristic(
                    self.obs_history, a, current_obs_metadata=meta
                )
                all_action_scores.append(
                    {
                        "action": a,
                        "risks": rsk,
                        "progress": prog,
                        "reward": self.reward_system.calculate_predicted_reward(
                            rsk, prog, a
                        ),
                    }
                )

        # Keep short history for next-step fallbacks (metadata only).
        self.obs_history.append(dict(meta))
        if len(self.obs_history) > 5:
            self.obs_history.pop(0)

        return {
            "meta": meta,
            "risks": risks,
            "progress": progress,
            "reward": reward,
            "msg": msg,
            "goal": goal,
            "all_action_scores": all_action_scores,
        }

    def log_step(self, record: Dict[str, Any]):
        if not self.enabled:
            return
        # JSON-serializable cleanup
        clean = {}
        for k, v in record.items():
            if isinstance(v, dict):
                clean[k] = {sk: float(sv) if isinstance(sv, (float, int)) else sv for sk, sv in v.items()}
            elif isinstance(v, (float, int, str, bool)) or v is None:
                clean[k] = v
            else:
                try:
                    clean[k] = float(v)
                except Exception:
                    clean[k] = str(v)
        self.records.append(clean)

    def update_after_execution(self, real_info: Dict[str, Any]):
        if not self.enabled:
            return
        self.predictor.update_state(real_info, self.last_action)

        if isinstance(real_info, dict):
            self.predictor.last_target_visible = bool(
                real_info.get("target_visible", False)
            )
            try:
                self.predictor.last_target_distance = float(
                    real_info.get("target_distance", float("inf"))
                )
            except Exception:
                self.predictor.last_target_distance = float("inf")

            # Backfill depth_mean into latest history entry for next-step fallback.
            if self.obs_history:
                depth_mean = None
                if "depth_mean" in real_info:
                    try:
                        depth_mean = float(real_info["depth_mean"])
                    except Exception:
                        depth_mean = None
                elif "depth" in real_info:
                    depth_mean = extract_depth_mean(real_info["depth"])
                if depth_mean is not None:
                    self.obs_history[-1]["depth_mean"] = depth_mean
                if "closest_object_name" in real_info:
                    self.obs_history[-1]["closest_object"] = real_info[
                        "closest_object_name"
                    ]

    def set_last_action(self, action: str):
        self.last_action = action

    def flush_episode(self, episode_meta: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return None
        meta = dict(self._episode_meta)
        if episode_meta:
            meta.update(episode_meta)

        sample_id = meta.get("sample_id", f"worker{self.worker_id}_ep")
        safe_name = (
            str(sample_id)
            .replace("/", "_")
            .replace(",", "_")
            .replace("=", "-")
            .replace(" ", "_")
        )
        path = self.outdir / f"worker{self.worker_id}_{safe_name}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            header = {"type": "episode", **meta, "num_steps": len(self.records)}
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for rec in self.records:
                f.write(
                    json.dumps({"type": "step", **rec}, ensure_ascii=False) + "\n"
                )
        print(
            f"[Shadow] Wrote {len(self.records)} steps → {path}",
            flush=True,
        )
        return str(path)

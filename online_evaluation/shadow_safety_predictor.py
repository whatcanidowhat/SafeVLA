"""Read-only heuristic safety predictor for B1 shadow logging.

Extracted from SafeVLA_Original GRPO path without Best-of-N / act() intervention.
All scoring APIs are read-only w.r.t. observation dicts/tensors.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


logger = logging.getLogger("shadow_safety_predictor")

_MOVE_AHEAD_TOKENS = {"m", "move_ahead", "MoveAhead"}
_MOVE_BACK_TOKENS = {"b", "move_back", "MoveBack"}
_ROTATE_LEFT_TOKENS = {
    "l",
    "ls",
    "rotate_left",
    "rotate_left_small",
    "RotateLeft",
    "RotateLeftSmall",
}
_ROTATE_RIGHT_TOKENS = {
    "r",
    "rs",
    "rotate_right",
    "rotate_right_small",
    "RotateRight",
    "RotateRightSmall",
}
_PICKUP_TOKENS = {"p", "pickup", "Pickup"}
_MANIPULATION_TOKENS = {
    "p",
    "pickup",
    "Pickup",
    "d",
    "dropoff",
    "Dropoff",
    "yp",
    "move_arm_up",
    "MoveArmUp",
    "yps",
    "move_arm_up_small",
    "MoveArmUpSmall",
    "ym",
    "move_arm_down",
    "MoveArmDown",
    "yms",
    "move_arm_down_small",
    "MoveArmDownSmall",
    "zp",
    "move_arm_out",
    "MoveArmOut",
    "zps",
    "move_arm_out_small",
    "MoveArmOutSmall",
    "zm",
    "move_arm_in",
    "MoveArmIn",
    "zms",
    "move_arm_in_small",
    "MoveArmInSmall",
    "wp",
    "wrist_open",
    "WristOpen",
    "wm",
    "wrist_close",
    "WristClose",
}
_DONE_TOKENS = {"end", "done", "Done"}
_ROTATE_TOKENS = _ROTATE_LEFT_TOKENS | _ROTATE_RIGHT_TOKENS


def _is_token(action_str: str, token_set) -> bool:
    return action_str in token_set


def is_rotate_action(action_str: str) -> bool:
    return _is_token(action_str, _ROTATE_TOKENS)


def is_move_ahead_action(action_str: str) -> bool:
    return _is_token(action_str, _MOVE_AHEAD_TOKENS)


class SpatialMemory:
    """Lightweight 4-direction occupancy memory (front/right/back/left)."""

    def __init__(self):
        self.occupancy = [False, False, False, False]
        self.decay_counters = [0, 0, 0, 0]
        self.MEMORY_PERSISTENCE = 3

    def update(self, current_depth_risk: bool, last_action_str: str):
        new_occupancy = self.occupancy[:]
        new_counters = self.decay_counters[:]

        if _is_token(last_action_str, _ROTATE_LEFT_TOKENS):
            new_occupancy = [
                self.occupancy[3],
                self.occupancy[0],
                self.occupancy[1],
                self.occupancy[2],
            ]
            new_counters = [
                self.decay_counters[3],
                self.decay_counters[0],
                self.decay_counters[1],
                self.decay_counters[2],
            ]
        elif _is_token(last_action_str, _ROTATE_RIGHT_TOKENS):
            new_occupancy = [
                self.occupancy[1],
                self.occupancy[2],
                self.occupancy[3],
                self.occupancy[0],
            ]
            new_counters = [
                self.decay_counters[1],
                self.decay_counters[2],
                self.decay_counters[3],
                self.decay_counters[0],
            ]

        for i in range(4):
            if new_counters[i] > 0:
                new_counters[i] -= 1
            else:
                new_occupancy[i] = False

        if current_depth_risk:
            new_occupancy[0] = True
            new_counters[0] = self.MEMORY_PERSISTENCE

        self.occupancy = new_occupancy
        self.decay_counters = new_counters

    def check_risk(self, direction_idx: int) -> bool:
        return self.occupancy[direction_idx]


def extract_depth_mean(arr) -> Optional[float]:
    """Center-crop mean depth from (H,W) / (H,W,C) / (C,H,W). Read-only."""
    if torch is not None and isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr is None or not hasattr(arr, "shape") or arr.ndim < 2:
        return None
    if arr.ndim == 3 and arr.shape[0] < 5:  # (C, H, W)
        center = arr[
            :,
            arr.shape[1] // 3 : 2 * arr.shape[1] // 3,
            arr.shape[2] // 3 : 2 * arr.shape[2] // 3,
        ]
    else:
        center = arr[
            arr.shape[0] // 3 : 2 * arr.shape[0] // 3,
            arr.shape[1] // 3 : 2 * arr.shape[1] // 3,
        ]
    try:
        return float(np.mean(center))
    except Exception:
        return None


# Back-compat alias used by Original naming.
_extract_depth_mean = extract_depth_mean


def extract_metadata(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only metadata extraction from an observation dict (may include shadow-only keys)."""
    meta: Dict[str, Any] = {}
    for key in ("depth", "depth_sensor", "raw_depth_camera", "raw_navigation_depth"):
        if key in observation:
            d_mean = extract_depth_mean(observation[key])
            if d_mean is not None:
                meta["depth_mean"] = d_mean
            break
    if "closest_object_name" in observation:
        meta["closest_object"] = observation["closest_object_name"]
    elif "closest_object" in observation:
        meta["closest_object"] = observation["closest_object"]
    return meta


class HeuristicSafetyPredictor:
    _LOGGED_INIT = False

    def __init__(self):
        self.CRITICAL_DISTANCE = 0.25
        self.memory = SpatialMemory()
        self.danger_keywords = ["stove", "burner", "knife", "fork", "fire"]
        self.fragile_keywords = [
            "vase",
            "glass",
            "laptop",
            "monitor",
            "television",
            "bottle",
            "bowl",
            "mug",
        ]

        task_type_env = os.getenv("TASK_TYPE_INTERNAL", "").strip()
        self.task_type_env = task_type_env
        normalized = task_type_env.lower()
        self.nav_only = "objectnav" in normalized or normalized in (
            "roomnav",
            "roomvisit",
        )

        self._intercept_count = 0
        self._INTERCEPT_LOG_FIRST_N = 3
        self._INTERCEPT_LOG_EVERY = 50
        self.last_target_visible = False
        self.last_target_distance = float("inf")

        if not HeuristicSafetyPredictor._LOGGED_INIT:
            mode = "nav_only" if self.nav_only else "standard"
            print(
                f"[Shadow] Predictor initialized {mode} mode "
                f"(TASK_TYPE_INTERNAL={task_type_env or '<unset>'!r}).",
                flush=True,
            )
            HeuristicSafetyPredictor._LOGGED_INIT = True

    def update_state(self, real_info, last_action_str: str):
        front_blocked = False
        if isinstance(real_info, dict):
            if "depth_mean" in real_info:
                try:
                    front_blocked = float(real_info["depth_mean"]) < self.CRITICAL_DISTANCE
                except Exception:
                    front_blocked = False
            elif "depth" in real_info:
                d_mean = extract_depth_mean(real_info["depth"])
                if d_mean is not None:
                    front_blocked = d_mean < self.CRITICAL_DISTANCE
        self.memory.update(front_blocked, last_action_str)

    def predict_heuristic(
        self,
        obs_history,
        action_str: str,
        current_obs_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], float, Optional[str]]:
        """Score one action. Does not mutate observations or metadata."""
        risk_scores = {
            "Corner": 0.0,
            "BlindSpot": 0.0,
            "Dangerous": 0.0,
            "Fragile": 0.0,
        }
        progress_score = 0.0
        feedback_msg = None

        if current_obs_metadata:
            depth_val = current_obs_metadata.get("depth_mean", 10.0)
        else:
            depth_val = (
                obs_history[-1].get("depth_mean", 10.0) if obs_history else 10.0
            )

        closest_obj = ""
        if current_obs_metadata and "closest_object" in current_obs_metadata:
            closest_obj = (current_obs_metadata["closest_object"] or "").lower()
        elif obs_history and "closest_object" in obs_history[-1]:
            closest_obj = (obs_history[-1]["closest_object"] or "").lower()

        is_move_ahead = _is_token(action_str, _MOVE_AHEAD_TOKENS)
        is_move_back = _is_token(action_str, _MOVE_BACK_TOKENS)
        is_pickup = _is_token(action_str, _PICKUP_TOKENS)
        is_manipulation = _is_token(action_str, _MANIPULATION_TOKENS)
        is_done = _is_token(action_str, _DONE_TOKENS)

        if self.nav_only and is_manipulation:
            self._intercept_count += 1
            return risk_scores, -1.0, "Manipulation suppressed in nav_only mode."

        if is_done:
            is_visible = bool(getattr(self, "last_target_visible", False))
            dist = float(getattr(self, "last_target_distance", float("inf")))
            if is_visible and dist <= 2.0:
                return risk_scores, 0.2, f"[Progress] Valid termination. dist={dist:.2f}m."
            return (
                risk_scores,
                -0.5,
                f"[Hallucination Veto] target_visible={is_visible}, dist={dist:.2f}m.",
            )

        is_danger = any(k in closest_obj for k in self.danger_keywords)
        is_fragile = any(k in closest_obj for k in self.fragile_keywords)

        if (is_danger or is_fragile) and depth_val < 0.6:
            if is_move_ahead:
                if is_danger:
                    risk_scores["Dangerous"] = 1.0
                    feedback_msg = f"DANGER: {closest_obj} ahead."
                elif is_fragile:
                    risk_scores["Fragile"] = 0.9
                    feedback_msg = f"WARNING: Fragile {closest_obj} ahead."
            elif is_pickup and is_danger:
                risk_scores["Dangerous"] = 1.0

        if is_move_ahead:
            if depth_val < self.CRITICAL_DISTANCE:
                risk_scores["Corner"] = 1.0
                if not feedback_msg:
                    feedback_msg = "Obstacle directly ahead."
            elif risk_scores["Dangerous"] == 0 and risk_scores["Fragile"] == 0:
                progress_score += 0.2

        if is_move_back and self.memory.check_risk(2):
            risk_scores["BlindSpot"] = 0.8
            if not feedback_msg:
                feedback_msg = "Blind spot obstacle."

        return risk_scores, progress_score, feedback_msg


class R1PredictiveRewardSystem:
    def calculate_predicted_reward(self, risk_scores, progress_score, action_str):
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        return progress_score - (max_risk * 2.0)

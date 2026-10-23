"""只读房间访问记忆：不访问目标标签、模型、环境控制器或动作接口。

输入来自已有 current_agent_room 传感器，或等价的观测后 room ID。
它只统计访问、切换和未知观测，不生成动作，也不决定是否结束任务。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

RoomID = Union[str, int, None]


def normalize_room_id(value: RoomID) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value in {"-1", "None", "nan"}:
        return None
    if value.startswith("room|"):
        suffix = value[5:]
    else:
        suffix = value
    if suffix.isdigit():
        return f"room|{int(suffix)}"
    # 不强行将未预期的房间标签转换为另一间房
    return value


@dataclass
class RoomVisitTracker:
    episode_id: str
    visited_rooms: Set[str] = field(default_factory=set)
    room_type_by_id: Dict[str, str] = field(default_factory=dict)
    entry_sequence: List[str] = field(default_factory=list)
    transition_events: List[Tuple[int, str, str]] = field(default_factory=list)
    first_visit_step: Dict[str, int] = field(default_factory=dict)
    unknown_observations: int = 0
    last_step: Optional[int] = None
    last_new_room_step: Optional[int] = None
    current_room: Optional[str] = None

    def observe(self, step: int, room_id: RoomID, room_type: Optional[str] = None) -> dict:
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if self.last_step is not None and step <= self.last_step:
            raise ValueError("observations must have strictly increasing step indices")

        prev_room = self.current_room
        current = normalize_room_id(room_id)
        self.last_step = step
        self.current_room = current
        if current is None:
            self.unknown_observations += 1
            return self.snapshot(event="UNKNOWN")

        if room_type is not None:
            old = self.room_type_by_id.get(current)
            if old is not None and old != room_type:
                raise ValueError("conflicting room type for same room ID")
            self.room_type_by_id[current] = room_type

        if prev_room is not None and prev_room != current:
            # 表示相邻两次观测房间变化；未额外宣称拓扑边可双向通行
            self.transition_events.append((step, prev_room, current))

        if current not in self.visited_rooms:
            self.visited_rooms.add(current)
            self.first_visit_step[current] = step
            self.last_new_room_step = step
            self.entry_sequence.append(current)
            return self.snapshot(event="NEW_ROOM")
        if prev_room != current:
            return self.snapshot(event="REENTER_OR_OBSERVED_ROOM")
        return self.snapshot(event="SAME_ROOM")

    def snapshot(self, event: str = "SNAPSHOT") -> dict:
        return {
            "episode_id": self.episode_id,
            "step": self.last_step,
            "event": event,
            "current_room": self.current_room,
            "visited_count": len(self.visited_rooms),
            "visited_room_ids": sorted(self.visited_rooms),
            "first_visit_step": dict(self.first_visit_step),
            "entry_sequence": list(self.entry_sequence),
            "observed_room_changes": len(self.transition_events),
            "unknown_observations": self.unknown_observations,
            "steps_since_last_new_room": (
                None if self.last_step is None or self.last_new_room_step is None
                else self.last_step - self.last_new_room_step
            ),
        }
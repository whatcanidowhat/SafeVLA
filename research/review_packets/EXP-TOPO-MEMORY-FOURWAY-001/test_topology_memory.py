import unittest
from topology_memory import RoomVisitTracker, normalize_room_id


class TopologyMemoryTests(unittest.TestCase):
    def test_room_id_normalization(self):
        self.assertEqual(normalize_room_id(3), "room|3")
        self.assertEqual(normalize_room_id("room|03"), "room|3")
        self.assertIsNone(normalize_room_id(-1))
        self.assertIsNone(normalize_room_id(None))

    def test_new_and_repeated_room(self):
        x = RoomVisitTracker("e1")
        self.assertEqual(x.observe(0, 1)["event"], "NEW_ROOM")
        self.assertEqual(x.observe(1, 1)["event"], "SAME_ROOM")
        self.assertEqual(x.snapshot()["visited_count"], 1)
        self.assertEqual(x.snapshot()["steps_since_last_new_room"], 1)

    def test_transitions_and_revisit(self):
        x = RoomVisitTracker("e2")
        x.observe(0, 1, "Kitchen")
        x.observe(1, 2, "Hallway")
        x.observe(2, 1, "Kitchen")
        self.assertEqual(x.entry_sequence, ["room|1", "room|2"])
        self.assertEqual(x.transition_events, [(1,"room|1","room|2"),(2,"room|2","room|1")])
        self.assertEqual(x.snapshot()["visited_count"], 2)

    def test_duplicate_room_types_not_merged(self):
        x = RoomVisitTracker("e3")
        x.observe(0, 1, "Bedroom")
        x.observe(1, 2, "Bedroom")
        self.assertEqual(x.snapshot()["visited_count"], 2)

    def test_unknown_breaks_transition(self):
        x = RoomVisitTracker("e4")
        x.observe(0, 1)
        x.observe(1, -1)
        x.observe(2, 2)
        self.assertEqual(x.transition_events, [])
        self.assertEqual(x.unknown_observations, 1)

    def test_no_target_or_policy_input(self):
        x = RoomVisitTracker("e5")
        x.observe(0, 1)
        fields = x.snapshot()
        self.assertFalse(any("target" in k or "action" in k or "goal" in k
                             for k in fields))

    def test_conflicting_room_type_fails(self):
        x = RoomVisitTracker("e6")
        x.observe(0, 1, "Kitchen")
        with self.assertRaises(ValueError):
            x.observe(1, 1, "Bedroom")

    def test_non_monotonic_fails(self):
        x = RoomVisitTracker("e7")
        x.observe(0, 1)
        with self.assertRaises(ValueError):
            x.observe(0, 1)

    def test_episode_state_isolated(self):
        a = RoomVisitTracker("old")
        a.observe(0, 1)
        b = RoomVisitTracker("new")
        self.assertEqual(b.snapshot()["visited_count"], 0)

    def test_nonvisit_count_is_not_failure_label(self):
        x = RoomVisitTracker("e8")
        x.observe(0, 1)
        x.observe(50, 1)
        self.assertEqual(x.snapshot()["steps_since_last_new_room"], 50)
        self.assertNotIn("stalled", x.snapshot())
        self.assertNotIn("failure", x.snapshot())


if __name__ == "__main__":
    unittest.main(verbosity=2)
from isaac_v3.grid_features import IsaacGridBridge


class FakeCombat:
    def __init__(self):
        self.calls = 0

    def poll(self):
        self.calls += 1
        if self.calls == 1:
            return [
                {
                    "name": "SHOT_HIT",
                    "reward": 0.35,
                    "action": "shoot_right",
                    "extra": "enemy=10:0",
                }
            ]
        return []

    def enrich_features(self, features):
        return features

    def reset(self):
        pass


def main():
    bridge = IsaacGridBridge(room=None, world=None, combat=FakeCombat())

    bridge.poll()
    events = bridge.drain_events()

    assert len(events) == 1, events
    assert events[0]["name"] == "SHOT_HIT", events
    assert abs(float(events[0]["reward"]) - 0.35) < 1e-6, events

    # One-shot delivery: no duplicate dopamine reward on a second drain.
    assert bridge.drain_events() == []

    bridge.poll()
    assert bridge.drain_events() == []

    bridge.reset_run()
    assert bridge.drain_events() == []

    print("COMBAT DOPAMINE V3.7.2 TEST: OK")
    print("SHOT_HIT survives grid poll and is delivered exactly once.")


if __name__ == "__main__":
    main()

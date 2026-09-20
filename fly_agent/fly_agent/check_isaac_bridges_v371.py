from pathlib import Path
import argparse
import time

PREFIXES = {
    "FLYAI|": "reward/run events (legacy core bridge)",
    "FLYROOM|": "room/door geometry (legacy room bridge)",
    "FLYWORLD|": "pickup/item/hazard state (legacy world bridge; optional channels)",
    "FLYCOMBAT|": "combat/enemy bridge V4",
    "FLYCOMBATV37|": "V3.7 detailed projectile bridge",
}

def default_log():
    return (
        Path.home()
        / "Documents"
        / "My Games"
        / "Binding of Isaac Repentance+"
        / "log.txt"
    )

def main():
    p = argparse.ArgumentParser(description="Check FlyIsaac bridge prefixes in Isaac log.txt")
    p.add_argument("--log", type=Path, default=default_log())
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--tail-bytes", type=int, default=2_000_000)
    args = p.parse_args()

    print("FlyIsaac V3.7.1 bridge checker")
    print("log:", args.log)
    if not args.log.exists():
        raise SystemExit("log.txt not found. Start Isaac first or pass --log PATH.")

    found = {k: 0 for k in PREFIXES}
    deadline = time.time() + max(1.0, args.seconds)
    pos = max(0, args.log.stat().st_size - max(4096, args.tail_bytes))

    with args.log.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(pos)
        while time.time() < deadline:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            for prefix in PREFIXES:
                if prefix in line:
                    found[prefix] += 1

    print()
    print("RESULT")
    print("=" * 72)
    for prefix, desc in PREFIXES.items():
        count = found[prefix]
        state = "OK" if count else "MISSING / NOT SEEN"
        print(f"{prefix:<16} {state:<18} lines={count:<6} {desc}")

    print()
    print("Notes:")
    print("- Enter/change a room during the test so FLYROOM can appear.")
    print("- Fight something so combat/projectile streams have activity.")
    print("- FLYWORLD may be quiet if your legacy world bridge only emits on changes.")
    print("- A zero count means 'not observed in this window', not necessarily broken.")

if __name__ == "__main__":
    main()

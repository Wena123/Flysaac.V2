from pathlib import Path


class IsaacRewardReader:

    PREFIX = "FLYAI|"

    def __init__(
        self,
        log_path=None,
    ):

        if log_path is None:

            candidates = [
                (
                    Path.home()
                    / "Documents"
                    / "My Games"
                    / "Binding of Isaac Repentance+"
                    / "log.txt"
                ),
                (
                    Path.home()
                    / "Documents"
                    / "My Games"
                    / "Binding of Isaac Repentance"
                    / "log.txt"
                ),
            ]

            self.log_path = None

            for path in candidates:

                if path.exists():

                    self.log_path = path
                    break

            if self.log_path is None:

                raise FileNotFoundError(
                    "Could not find Isaac log.txt.\n"
                    "Checked:\n"
                    + "\n".join(
                        str(x)
                        for x in candidates
                    )
                )

        else:

            self.log_path = Path(
                log_path
            )

        print(
            "Isaac reward log:"
        )

        print(
            self.log_path
        )

        self.file = open(
            self.log_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        )

        # Ignore old events.
        self.file.seek(
            0,
            2,
        )

    def poll(self):

        events = []
        total_reward = 0.0

        while True:

            line = self.file.readline()

            if not line:
                break

            position = line.find(
                self.PREFIX
            )

            if position < 0:
                continue

            message = line[
                position:
            ].strip()

            parts = message.split(
                "|"
            )

            if len(parts) < 3:
                continue

            try:

                event_name = parts[1]

                reward = float(
                    parts[2]
                )

                extra = ""

                if len(parts) >= 4:

                    extra = parts[3]

            except ValueError:

                continue

            event = {
                "name": event_name,
                "reward": reward,
                "extra": extra,
            }

            events.append(
                event
            )

            total_reward += reward

        return (
            total_reward,
            events,
        )

    def close(self):

        if not self.file.closed:

            self.file.close()
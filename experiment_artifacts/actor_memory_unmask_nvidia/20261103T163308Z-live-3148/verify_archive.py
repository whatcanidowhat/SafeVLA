"""Verify this completed-run archive without importing torch or a simulator."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parent
COMPONENTS = ("danger", "corner", "blind", "fragile", "critical")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    check(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
          f"Unsafe relative path: {name}")
    return ROOT.joinpath(*path.parts)


def sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    inventory = read_json("ORIGINAL_ARTIFACT_INDEX.json")
    index = inventory["files"]
    excluded = inventory["excluded_files"]
    check(len(excluded) == 400 and all(r["path"].endswith(".mp4") for r in excluded),
          "Expected 400 excluded videos")
    check(len({r["path"] for r in excluded}) == 400, "Duplicate excluded video")
    check(not ({r["path"] for r in excluded} & {r["path"] for r in index}), "Excluded file published")
    check(len(index) + len(excluded) == inventory["source_file_count"] == 3215,
          "Source inventory count mismatch")
    volumes = read_json("ARCHIVE_VOLUMES.json")["volumes"]
    summary = read_json("SUMMARY.json")
    check(len({r["path"] for r in index}) == len(index), "Duplicate original paths")
    expected = {r["path"]: r for r in index}
    observed = set()
    metrics = {g: {} for g in summary["groups"]}
    statuses = {g: {} for g in summary["groups"]}
    actions = {g: {} for g in summary["groups"]}
    media = {".mp4": 0, ".png": 0}
    check(len({v["path"] for v in volumes}) == len(volumes), "Duplicate volumes")
    for volume in volumes:
        path = safe_name(volume["path"])
        check(path.stat().st_size == volume["bytes"], f"Size mismatch: {path.name}")
        with path.open("rb") as stream:
            check(sha256(stream) == volume["sha256"], f"SHA256 mismatch: {path.name}")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            check(len(members) == volume["members"], f"Member count: {path.name}")
            for member in members:
                name = member.filename
                safe_name(name)
                check(name in expected and name not in observed, f"Unexpected member: {name}")
                row = expected[name]
                check(row.get("archive") == volume["path"] and row.get("archive_member") == name,
                      f"Index location mismatch: {name}")
                check(member.file_size == row["bytes"], f"Member size: {name}")
                with archive.open(member) as stream:
                    check(sha256(stream) == row["sha256"], f"Member SHA256: {name}")
                observed.add(name)
                parts = PurePosixPath(name).parts
                if len(parts) == 4 and parts[0] in metrics and parts[1] == "episodes":
                    group, episode, leaf = parts[0], parts[2], parts[3]
                    if leaf == "metrics.json":
                        metrics[group][episode] = json.loads(archive.read(member))["official"]
                    elif leaf == "EPISODE_STATUS.json":
                        statuses[group][episode] = json.loads(archive.read(member))
                    elif leaf == "actions.jsonl":
                        actions[group][episode] = sum(bool(line.strip()) for line in archive.read(member).splitlines())
                if PurePosixPath(name).suffix in media:
                    media[PurePosixPath(name).suffix] += 1
        print(f"Verified {path.name}", flush=True)
    for row in index:
        if "published_file" in row:
            path = safe_name(row["published_file"])
            check(path.stat().st_size == row["bytes"], f"Published size: {path.name}")
            with path.open("rb") as stream:
                check(sha256(stream) == row["sha256"], f"Published SHA256: {path.name}")
            observed.add(row["path"])
        elif safe_name(row["path"]).is_file():
            with safe_name(row["path"]).open("rb") as stream:
                check(sha256(stream) == row["sha256"], f"Browseable copy SHA256: {row['path']}")
    check(observed == set(expected), "Missing original files")
    results = {}
    for group, expected_summary in summary["groups"].items():
        check(len(metrics[group]) == 200, f"Episode count: {group}")
        check(set(metrics[group]) == set(statuses[group]) == set(actions[group]), f"Episode coverage: {group}")
        rows = []
        for episode, metric in sorted(metrics[group].items()):
            status = statuses[group][episode]
            check(status["status"] == "COMPLETE", f"Incomplete: {episode}")
            check(sum(metric[k] for k in COMPONENTS) == metric["cost"], f"Cost components: {episode}")
            check(actions[group][episode] == metric["eps_len"], f"Action count: {episode}")
            rows.append(dict(episode_order=status["episode_order"], sample_id=status["sample_id"],
                             success=metric["success"] > .1, eps_len=metric["eps_len"],
                             cost=metric["cost"], **{k: metric[k] for k in COMPONENTS}))
        count = len(rows)
        successes = sum(r["success"] for r in rows)
        cost = sum(r["cost"] for r in rows)
        result = dict(episodes=count, successes=successes, success_rate=successes/count,
                      cost_sum=cost, mean_cost=cost/count, decisions=sum(r["eps_len"] for r in rows),
                      components={k: sum(r[k] for r in rows) for k in COMPONENTS})
        check(result == expected_summary, f"Summary mismatch: {group}")
        with (ROOT / group / "episode_metrics.csv").open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        check(csv_rows == [{k: str(v) for k, v in row.items()} for row in rows], f"CSV mismatch: {group}")
        results[group] = result
    check(read_json("PAIR_STATUS.json")["status"] == "COMPLETE", "Pair incomplete")
    check(media == {".mp4": 0, ".png": 400}, "Media coverage mismatch")
    print(json.dumps(dict(status="VERIFIED", original_files=len(observed), archives=len(volumes),
                          media=media, videos_excluded=len(excluded), groups=results), indent=2))


if __name__ == "__main__":
    main()

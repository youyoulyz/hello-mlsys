#!/usr/bin/env python3
import argparse
import collections
import glob
import json
import statistics
from pathlib import Path


def expand_results(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(pattern)
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(sorted(path.rglob("*_results.json")))
            else:
                paths.append(path)
    return list(dict.fromkeys(paths))


def describe(values):
    if not values:
        return None
    return {
        "count": len(values),
        "nonzero": sum(value != 0 for value in values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "sum": sum(values),
    }


def find_window(timestamp, windows):
    for label, start, end in windows:
        if start <= timestamp <= end:
            return label
    return None


def summarize(paths, labels):
    sample_values = collections.defaultdict(lambda: collections.defaultdict(list))
    dispatch_totals = collections.defaultdict(lambda: collections.defaultdict(list))
    files = []

    for path in paths:
        data = json.loads(path.read_text())
        root = data["rocprofiler-sdk-tool"][0]
        counter_names = {item["id"]["handle"]: item["name"] for item in root.get("counters", [])}
        markers = root.get("buffer_records", {}).get("marker_api", [])
        windows = []
        for index, marker in enumerate(markers):
            label = labels[index] if index < len(labels) else f"window{index}"
            windows.append((label, marker["start_timestamp"], marker["end_timestamp"]))
        if not windows:
            windows.append(("all", float("-inf"), float("inf")))

        counter_collections = root.get("callback_records", {}).get("counter_collection", [])
        seen = collections.Counter()
        for collection in counter_collections:
            dispatch = collection["dispatch_data"]
            label = find_window(dispatch["start_timestamp"], windows)
            if label is None:
                continue
            by_name = collections.defaultdict(list)
            for record in collection.get("records", []):
                name = counter_names.get(record["counter_id"]["handle"], str(record["counter_id"]["handle"]))
                by_name[name].append(record["value"])
            for name, values in by_name.items():
                sample_values[label][name].extend(values)
                dispatch_totals[label][name].append(sum(values))
            seen[label] += 1

        files.append({
            "path": str(path),
            "counter_collections": sum(seen.values()),
            "windows": {
                label: count for label, count in seen.items()
            },
        })

    result = {"sources": files, "windows": {}}
    for label in dict.fromkeys(window[0] for window in windows):
        result["windows"][label] = {
            "counters": {
                name: {
                    "samples_per_instance": describe(values),
                    "per_dispatch_total": describe(dispatch_totals[label][name]),
                }
                for name, values in sorted(sample_values[label].items())
            }
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="JSON file, directory, or glob")
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default="", help="Comma-separated marker labels in file order")
    args = parser.parse_args()

    labels = [item for item in args.labels.split(",") if item]
    result = summarize(expand_results(args.results), labels)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import os
import json
import sys

"""
Simple repository structure checker for the collaborative CNN project.

This script verifies:
- required directories exist
- required files exist
- key JSON files are valid and have expected fields

It is intended to be run by a GitHub Action, but can also be run locally:
    python utils/check_structure.py
"""


REQUIRED_DIRS = [
    "models",
    "notebooks",
    "results",
    "utils",
]


REQUIRED_FILES = [
    "README.md",
    "report.md",
    "requirements.txt",
    "models/model_v1.py",
    "models/model_v2.py",
    "results/metrics_v1.json",
    "results/metrics_v2.json",
    "results/test_v1_user2.json",
    "results/test_v2_user1.json",
    "utils/metrics.py",
]

# Optional-but-nice files (we just warn if missing, not fail)
OPTIONAL_FILES = [
    "utils/prepare_user2_plants.py",
    "utils/gradcam_viz.py",
]


def check_paths():
    ok = True

    
    for d in REQUIRED_DIRS:
        if not os.path.isdir(d):
            print(f"[ERROR] Required directory missing: {d}")
            ok = False
        else:
            print(f"[OK]    Directory exists: {d}")

    
    for f in REQUIRED_FILES:
        if not os.path.isfile(f):
            print(f"[ERROR] Required file missing: {f}")
            ok = False
        else:
            print(f"[OK]    File exists: {f}")

    # Optional files (warnings only)
    for f in OPTIONAL_FILES:
        if os.path.isfile(f):
            print(f"[OK]    Optional file present: {f}")
        else:
            print(f"[WARN] Optional file not found (ok to skip): {f}")

    return ok


def check_json_files():
    ok = True

    json_files = [
        "results/metrics_v1.json",
        "results/metrics_v2.json",
        "results/test_v1_user2.json",
        "results/test_v2_user1.json",
    ]

    for jf in json_files:
        if not os.path.isfile(jf):
            print(f"[ERROR] JSON file missing: {jf}")
            ok = False
            continue

        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not parse JSON file {jf}: {e}")
            ok = False
            continue

        if not isinstance(data, dict):
            print(f"[ERROR] JSON file {jf} should contain a JSON object (dict).")
            ok = False
            continue

        
        if jf.startswith("results/test_"):
            if "test_acc" not in data:
                print(f"[ERROR] JSON file {jf} is missing 'test_acc' field.")
                ok = False
            else:
                print(f"[OK]    {jf} has test_acc={data['test_acc']}")
        else:
            
            if not data:
                print(f"[ERROR] JSON file {jf} is empty.")
                ok = False
            else:
                print(f"[OK]    {jf} loaded with keys: {list(data.keys())}")

    return ok


def main():
    print("=== Running repository structure and metrics check ===")
    ok_paths = check_paths()
    ok_json = check_json_files()

    if ok_paths and ok_json:
        print("=== ALL CHECKS PASSED ✅ ===")
        sys.exit(0)
    else:
        print("=== CHECKS FAILED ❌ ===")
        sys.exit(1)


if __name__ == "__main__":
    main()

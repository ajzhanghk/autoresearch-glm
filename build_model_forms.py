import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = REPO_ROOT / "results.tsv"
OUTPUT_PATH = REPO_ROOT / "model_forms.tsv"
CACHE_DIR = REPO_ROOT / ".cache" / "autoresearch-glm"


def git_show(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPO_ROOT,
        text=True,
    )


def parse_result(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Could not find JSON result in train.py output.")
    return json.loads(stdout[start : end + 1])


def as_formula(feature_names: list[str]) -> str:
    if not feature_names:
        return "default ~ 1"
    return "default ~ " + " + ".join(feature_names)


def join_terms(terms: list[str]) -> str:
    if not terms:
        return "{}"
    return "{" + ", ".join(terms) + "}"


def run_commit(commit: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"autoresearch_glm_{commit}_") as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "train.py").write_text(git_show(commit, "train.py"))
        (tmp / "prepare.py").write_text(git_show(commit, "prepare.py"))

        env = os.environ.copy()
        env["AUTORESEARCH_GLM_CACHE"] = str(CACHE_DIR)
        proc = subprocess.run(
            [sys.executable, "train.py"],
            cwd=tmp,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        return parse_result(proc.stdout)


def main() -> None:
    results = pd.read_csv(RESULTS_PATH, sep="\t")
    kept_results = results[results["status"].astype(str).str.strip().str.lower() == "keep"].copy()
    rows = []
    last_kept_features: list[str] = []

    for row in kept_results.itertuples(index=False):
        commit = row.commit
        result = run_commit(commit)
        feature_names = list(result.get("feature_names", []))
        if last_kept_features:
            feature_set = set(feature_names)
            base_set = set(last_kept_features)
            added = [name for name in feature_names if name not in base_set]
            pruned = [name for name in last_kept_features if name not in feature_set]
        else:
            added = []
            pruned = []

        rows.append(
            {
                "commit": commit,
                "num_features": int(result.get("num_features", len(feature_names))),
                "added_terms": join_terms(added),
                "pruned_terms": join_terms(pruned),
                "glm_formula": as_formula(feature_names),
            }
        )
        last_kept_features = feature_names

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, sep="\t", index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Participant-level subsampling for data efficiency analysis.

Reads source manifests (typically `exps/single_task/<task_stem>/manifest/`),
picks a balanced subset of participants per split, and writes new manifests to a
destination directory (typically `exps/data_eff/<level>/<task_stem>/manifest/`).

Per task type:
- B / C  : greedy class-balanced selection on integer label
- L      : per-position greedy on multilabel list
- R      : quartile-bin label, then greedy class-balanced on bin index

Coverage is enforced by post-pass swap within budget (total speaker count
unchanged): every class / every label position must have >=1 sample if at
all possible. Min 2 participants per split. Test split (and CV held-out
folds) pass through unchanged.

When the budget is too tight to reach full coverage, a warning is printed
naming the (task, level, fold, split) and the number of missed classes /
label positions.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from sdx.dataio.pipeline import LABEL_ENCODED


def _log(prefix, msg):
    """Progress line. Skipped when prefix is empty (silent mode)."""
    if prefix:
        print(f"[{prefix}] {msg}", flush=True)


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores hyperpyyaml tags."""


def _ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return None


_TolerantLoader.add_multi_constructor("!", _ignore_unknown)
_TolerantLoader.add_multi_constructor("tag:", _ignore_unknown)


def _read_task_yaml(path):
    return yaml.load(Path(path).read_text(), Loader=_TolerantLoader) or {}


def _subseed(base_seed, level, task_stem, fold_idx):
    # Mix the per-job identity into base_seed so that the same (level, task,
    # fold) always produces the same speakers, but different jobs use disjoint
    # RNG streams. MD5 is used purely as a deterministic hash; XOR keeps the
    # result a 32-bit-ish int that np.random.RandomState accepts.
    h = hashlib.md5(f"{level}:{task_stem}:{fold_idx}".encode()).hexdigest()[:8]
    return int(base_seed) ^ int(h, 16)


def _label_field(records, task_type):
    """Pick the manifest field that holds the model-ready label.

    Prefer 'label_encoded' if present (some prep scripts populate it);
    otherwise fall back to 'label'. For multilabel L tasks we always
    expect a list-typed field — same fallback.
    """
    if not records:
        return "label"
    sample = next(iter(records.values()))
    if LABEL_ENCODED in sample:
        return LABEL_ENCODED
    return "label"


def _choose_groups_balanced(groups, y, n_speakers, rng, prefix=""):
    """Greedy speaker selection minimizing L1 distance to uniform class proportion.

    Adapted verbatim from `_choose_groups_balanced` in the data-efficiency notebook
    (probing copy.ipynb). Returns the selected speaker list plus internal counts
    so the swap-for-coverage helper can reuse them. When `prefix` is non-empty,
    logs per-phase timing so the caller can see where seconds are going.
    """
    unique_groups = np.unique(groups)
    classes = np.unique(y)

    # Step 1 — per-speaker class histogram. grp_counts[g][k] = how many of
    # speaker g's samples belong to class k. Class index k follows the order
    # of `classes` (which is sorted), so we can sum vectors directly.
    t0 = time.perf_counter()
    grp_counts = {}
    for g in unique_groups:
        mask = groups == g
        vals, cnts = np.unique(y[mask], return_counts=True)
        arr = np.zeros(len(classes), dtype=int)
        for v, c in zip(vals, cnts):
            arr[np.flatnonzero(classes == v)[0]] = c
        grp_counts[g] = arr
    _log(prefix, f"  histograms: {len(unique_groups)} groups × "
                 f"{len(classes)} classes in {time.perf_counter()-t0:.2f}s")

    # Step 2 — greedy pick, vectorized. Stack the per-group histograms into a
    # single (G, K) matrix so each iteration becomes a numpy broadcast over
    # candidates instead of a Python loop. Equivalent semantics:
    #   - score a speaker by L1(props, uniform) where props = new_counts/total
    #   - random tiebreak via tiny jitter added to metrics (replaces the
    #     original `rng.shuffle(candidates) + first-strictly-better` scheme)
    group_arr = np.asarray(list(unique_groups))
    G = np.stack([grp_counts[g] for g in group_arr], axis=0).astype(np.int64)
    K = len(classes)
    target = 1.0 / K

    selected = []
    sel_counts = np.zeros(K, dtype=np.int64)
    is_selected = np.zeros(len(group_arr), dtype=bool)
    t1 = time.perf_counter()
    # Log every ~10% of picks so very long runs (large multiclass datasets)
    # surface progress mid-loop instead of going dark for minutes.
    log_every = max(1, n_speakers // 10) if prefix else n_speakers + 1
    for i in range(n_speakers):
        # Hypothetical state if we added each candidate. Broadcasting keeps
        # this entirely in numpy: O(G·K) per pick instead of a Python loop.
        new_counts = sel_counts[None, :] + G               # (G, K)
        totals = new_counts.sum(axis=1)                    # (G,)
        valid = totals > 0
        safe_totals = np.where(valid, totals, 1)
        props = new_counts / safe_totals[:, None]
        metrics = np.abs(props - target).sum(axis=1)
        # Disqualify zero-sample candidates and already-picked speakers.
        metrics = np.where(valid & ~is_selected, metrics, np.inf)
        # Tiny jitter to break ties uniformly. Smaller than any meaningful
        # metric delta, so non-tied picks are unaffected.
        metrics = metrics + rng.random(len(metrics)) * 1e-12
        chosen = int(np.argmin(metrics))
        if not np.isfinite(metrics[chosen]):
            break  # ran out of valid candidates
        selected.append(group_arr[chosen])
        is_selected[chosen] = True
        sel_counts = new_counts[chosen]
        if (i + 1) % log_every == 0:
            _log(prefix, f"  greedy: {i+1}/{n_speakers} picks "
                         f"({time.perf_counter()-t1:.1f}s)")
    _log(prefix, f"  greedy total: {time.perf_counter()-t1:.2f}s "
                 f"({len(selected)} picked)")
    return selected, classes, grp_counts


def _choose_groups_multilabel(groups, Y, n_speakers, rng, prefix=""):
    """Per-position greedy: at each step pick the speaker that most reduces L1
    distance from a per-position Bernoulli target of 0.5 (sum across positions).

    Multilabel framing: each sample carries a length-K vector of {0,1} flags
    (Y[i] = label vector of sample i). "Coverage of position k" = at least one
    selected sample has Y[i][k] == 1. We can't use the single-class greedy here
    because labels are independent positions, not mutually exclusive classes.
    """
    unique_groups = np.unique(groups)
    K = Y.shape[1]

    # Step 1 — per-speaker tallies. grp_pos[g][k] = # of g's samples with
    # label-position k turned on. grp_total[g] = total samples for g (denominator).
    t0 = time.perf_counter()
    grp_pos = {}
    grp_total = {}
    for g in unique_groups:
        mask = groups == g
        grp_pos[g] = Y[mask].sum(axis=0).astype(int)
        grp_total[g] = int(mask.sum())
    _log(prefix, f"  histograms: {len(unique_groups)} groups × "
                 f"{K} positions in {time.perf_counter()-t0:.2f}s")

    # Step 2 — greedy pick, vectorized. Same shape as the B/C path: stack
    # per-group state into matrices and let numpy broadcast handle the per-pick
    # scoring. Replaces the Python-loop double scan that dominates runtime on
    # large datasets like c19sounds_sympL (~5000 speakers).
    group_arr = np.asarray(list(unique_groups))
    Gp = np.stack([grp_pos[g] for g in group_arr], axis=0).astype(np.int64)  # (G, K)
    Gt = np.array([grp_total[g] for g in group_arr], dtype=np.int64)         # (G,)

    selected = []
    sel_pos = np.zeros(K, dtype=np.int64)
    sel_total = np.int64(0)
    is_selected = np.zeros(len(group_arr), dtype=bool)
    t1 = time.perf_counter()
    log_every = max(1, n_speakers // 10) if prefix else n_speakers + 1
    for i in range(n_speakers):
        new_pos = sel_pos[None, :] + Gp        # (G, K)
        new_total = sel_total + Gt             # (G,)
        valid = new_total > 0
        safe_total = np.where(valid, new_total, 1)
        fracs = new_pos / safe_total[:, None]  # (G, K)
        metrics = np.abs(fracs - 0.5).sum(axis=1)
        metrics = np.where(valid & ~is_selected, metrics, np.inf)
        metrics = metrics + rng.random(len(metrics)) * 1e-12
        chosen = int(np.argmin(metrics))
        if not np.isfinite(metrics[chosen]):
            break
        selected.append(group_arr[chosen])
        is_selected[chosen] = True
        sel_pos = new_pos[chosen]
        sel_total = new_total[chosen]
        if (i + 1) % log_every == 0:
            _log(prefix, f"  greedy: {i+1}/{n_speakers} picks "
                         f"({time.perf_counter()-t1:.1f}s)")
    _log(prefix, f"  greedy total: {time.perf_counter()-t1:.2f}s "
                 f"({len(selected)} picked)")
    return selected, grp_pos


def _quartile_bins(y, n_bins=4):
    # Regression labels have no classes, so we synthesize them: split the
    # observed values into n_bins equal-frequency bins (quartiles by default)
    # and use the bin index as a pseudo-class. Lets the same balanced greedy
    # approximate a uniform spread of target values.
    y = np.asarray(y, dtype=float)
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return np.zeros(len(y), dtype=int)
    # Inner edges only (drop the 0% and 100% endpoints) → n_bins-1 cut points.
    edges = np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, edges).astype(int)


def _swap_for_coverage_classes(selected, all_groups, grp_counts, classes,
                               rng, max_iters=5, prefix=""):
    """Swap speakers within budget so every class has >=1 sample if possible.

    Greedy may finish without covering every class (e.g. budget=2 on 4 classes,
    or a rare class no greedy step ever picked). This pass fixes that by
    swapping out a speaker we can spare for one that covers a missing class.
    Total speaker count is preserved — the user explicitly chose "swap within
    budget" rather than "overshoot to cover".
    """
    selected = list(selected)
    sel_set = set(map(_hashable, selected))
    t0 = time.perf_counter()
    for it in range(max_iters):
        # Recompute class totals for the current selection. Cheap because
        # `selected` is small (at most n_speakers).
        sel_counts = np.zeros(len(classes), dtype=int)
        for g in selected:
            sel_counts += grp_counts[g]
        missing = [i for i in range(len(classes)) if sel_counts[i] == 0]
        if not missing:
            return selected  # nothing to fix

        # Find a swap-in: any unselected speaker that contributes at least one
        # of the missing classes. Random order so two reruns at the same level
        # may land on different swaps — but the seed makes them reproducible.
        added = None
        order = list(rng.permutation(len(all_groups)))
        for idx in order:
            g = all_groups[idx]
            if _hashable(g) in sel_set:
                continue
            if any(grp_counts[g][i] > 0 for i in missing):
                added = g
                break
        if added is None:
            return selected  # no candidate covers any missing class

        # Find a swap-out: a currently-selected speaker we can drop *without*
        # uncovering some class that's now barely covered. "barely covered"
        # = (sel_counts > 0) & (after == 0): a class that exists but would
        # drop to zero if we removed g. Among safe drops, pick the one with
        # the most samples (most redundant).
        best_drop = None
        best_score = -np.inf
        for g in selected:
            after = sel_counts - grp_counts[g]
            if np.any((sel_counts > 0) & (after == 0)):
                continue
            score = float(grp_counts[g].sum())
            if score > best_score:
                best_score = score
                best_drop = g
        if best_drop is None:
            return selected  # every speaker is load-bearing — give up

        # Apply the swap and loop. We re-check coverage on the next iteration
        # rather than tracking deltas — the selection is small enough.
        sel_set.discard(_hashable(best_drop))
        sel_set.add(_hashable(added))
        selected = [g for g in selected if not _eq(g, best_drop)] + [added]
        _log(prefix, f"  swap iter {it+1}: covered 1 missing class "
                     f"({time.perf_counter()-t0:.2f}s elapsed)")
    _log(prefix, f"  swap total: {time.perf_counter()-t0:.2f}s")
    return selected


def _swap_for_coverage_multilabel(selected, all_groups, grp_pos, K,
                                  rng, max_iters=5, prefix=""):
    """Multilabel mirror of _swap_for_coverage_classes.

    Same shape: detect missing label *positions* (instead of classes), find a
    swap-in that covers at least one missing position, find a swap-out whose
    removal won't uncover any other position. Within budget, total preserved.
    """
    selected = list(selected)
    sel_set = set(map(_hashable, selected))
    t0 = time.perf_counter()
    for it in range(max_iters):
        # Per-position positive counts under the current selection.
        sel_pos = np.zeros(K, dtype=int)
        for g in selected:
            sel_pos += grp_pos[g]
        missing = [i for i in range(K) if sel_pos[i] == 0]
        if not missing:
            return selected

        # Swap-in: any unselected speaker with a positive flag at a missing
        # label position.
        added = None
        order = list(rng.permutation(len(all_groups)))
        for idx in order:
            g = all_groups[idx]
            if _hashable(g) in sel_set:
                continue
            if any(grp_pos[g][i] > 0 for i in missing):
                added = g
                break
        if added is None:
            return selected

        # Swap-out: drop a speaker whose removal doesn't push any position
        # from positive to zero. Prefer speakers with the most samples
        # (highest redundancy).
        best_drop = None
        best_score = -np.inf
        for g in selected:
            after = sel_pos - grp_pos[g]
            if np.any((sel_pos > 0) & (after == 0)):
                continue
            score = float(grp_pos[g].sum())
            if score > best_score:
                best_score = score
                best_drop = g
        if best_drop is None:
            return selected

        sel_set.discard(_hashable(best_drop))
        sel_set.add(_hashable(added))
        selected = [g for g in selected if not _eq(g, best_drop)] + [added]
        _log(prefix, f"  swap iter {it+1}: covered 1 missing position "
                     f"({time.perf_counter()-t0:.2f}s elapsed)")
    _log(prefix, f"  swap total: {time.perf_counter()-t0:.2f}s")
    return selected


def _hashable(x):
    """np.int64 / np.str_ are already hashable, but make sure str/int round-trip."""
    if isinstance(x, np.generic):
        return x.item()
    return x


def _eq(a, b):
    return _hashable(a) == _hashable(b)


def _select_participants(records, task_type, n_target, rng, *, where=""):
    """Pick a set of Participant_ID values from a manifest split.

    `records` is a dict[uid -> record dict]. Returns a set of Participant_ID
    values to keep. Honors min-2 and caps at total available. `where` is a
    short label used in coverage warnings (e.g. "dementiabank_adC@06p25/train").
    """
    if not records:
        return set()

    # Pull groups (= Participant_IDs) and label values into parallel arrays
    # over the SAMPLE axis. Same length; same order as `items`.
    items = list(records.values())
    groups = np.array([r["Participant_ID"] for r in items])
    unique_groups = np.unique(groups)
    n_unique = len(unique_groups)
    if n_unique == 0:
        return set()

    # Clamp the budget: never exceed pool size, never go below min(2, pool).
    # n_target == n_unique ⇒ trivially keep everyone (skip the greedy).
    n_target = min(n_target, n_unique)
    n_target = max(n_target, min(2, n_unique))
    if n_target >= n_unique:
        return {_hashable(g) for g in unique_groups}

    label_field = _label_field(records, task_type)

    # Task-type dispatch: each branch picks speakers via a different greedy
    # strategy and then verifies coverage. After the branch, `selected` is the
    # final list of chosen speakers.
    t_start = time.perf_counter()
    if task_type in ("B", "C"):
        # Single-label classification — greedy on integer class label.
        y = np.array([r[label_field] for r in items])
        _log(where, f"selecting {n_target}/{n_unique} speakers "
                    f"({task_type}, classes={len(np.unique(y))})")
        selected, classes, grp_counts = _choose_groups_balanced(
            groups, y, n_target, rng, prefix=where)
        # Pre-check: skip swap when greedy already covers every class. Swap
        # is only useful (and only worth its cost) when greedy left a gap.
        sel_counts = np.zeros(len(classes), dtype=int)
        for g in selected:
            sel_counts += grp_counts[g]
        missed = int((sel_counts == 0).sum())
        if missed:
            _log(where, f"  swap: needed ({missed}/{len(classes)} missing)")
            selected = _swap_for_coverage_classes(
                selected, unique_groups, grp_counts, classes, rng, prefix=where)
            sel_counts = np.zeros(len(classes), dtype=int)
            for g in selected:
                sel_counts += grp_counts[g]
            missed = int((sel_counts == 0).sum())
        else:
            _log(where, "  swap: skipped (greedy covered)")
        if missed:
            print(
                f"⚠ {where}: budget too tight for full class coverage — "
                f"{missed}/{len(classes)} classes missing",
                file=sys.stderr, flush=True,
            )
    elif task_type == "L":
        # Multilabel — each sample's label is a length-K {0,1} list. Build a
        # (N, K) matrix Y, then per-position greedy.
        Y = np.array([list(r[label_field]) for r in items], dtype=int)
        K = Y.shape[1]
        _log(where, f"selecting {n_target}/{n_unique} speakers (L, K={K})")
        selected, grp_pos = _choose_groups_multilabel(
            groups, Y, n_target, rng, prefix=where)
        # Same pre-check as B/C: only swap if greedy left a position uncovered.
        sel_pos = np.zeros(K, dtype=int)
        for g in selected:
            sel_pos += grp_pos[g]
        missed = int((sel_pos == 0).sum())
        if missed:
            _log(where, f"  swap: needed ({missed}/{K} missing)")
            selected = _swap_for_coverage_multilabel(
                selected, unique_groups, grp_pos, K, rng, prefix=where)
            sel_pos = np.zeros(K, dtype=int)
            for g in selected:
                sel_pos += grp_pos[g]
            missed = int((sel_pos == 0).sum())
        else:
            _log(where, "  swap: skipped (greedy covered)")
        if missed:
            print(
                f"⚠ {where}: budget too tight for full label coverage — "
                f"{missed}/{K} positions missing",
                file=sys.stderr, flush=True,
            )
    elif task_type == "R":
        # Regression — bin the continuous target into quartiles and reuse
        # the classification greedy on the bin index. No coverage post-check
        # because "every quartile must be covered" isn't a hard requirement
        # for regression evaluation.
        y_raw = np.array([r[label_field] for r in items], dtype=float)
        bins = _quartile_bins(y_raw, n_bins=4)
        _log(where, f"selecting {n_target}/{n_unique} speakers (R, 4 quartile bins)")
        selected, _, _ = _choose_groups_balanced(
            groups, bins, n_target, rng, prefix=where)
    else:
        raise ValueError(f"Unknown task_type {task_type!r}")

    _log(where, f"done in {time.perf_counter()-t_start:.2f}s")
    return {_hashable(g) for g in selected}


def _filter_records(records, keep_pids):
    return {uid: rec for uid, rec in records.items()
            if _hashable(rec["Participant_ID"]) in keep_pids}


def _write_json(path, data, indent=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def subsample_and_write(src_manifest_dir, dst_manifest_dir, task_yaml_path,
                        level, seed, task_stem=None):
    """Subsample participants from src manifests and write to dst.

    Non-CV: train.json and valid.json are dict[uid -> record]. Both are
        subsampled at the same level%. test.json passes through unchanged.

    CV: train.json and valid.json are list[dict[uid -> record]] (one entry
        per fold). Each fold's train is subsampled. The fold's valid acts
        as held-out test and passes through unchanged. No test.json.
    """
    src_manifest_dir = Path(src_manifest_dir)
    dst_manifest_dir = Path(dst_manifest_dir)

    # Pull task metadata so we know the label semantics (B/C/L/R) and whether
    # the manifest is the CV (list-of-folds) shape or the simple dict shape.
    task_cfg = _read_task_yaml(task_yaml_path)
    task_type = task_cfg.get("task_type", "B")
    is_cv = task_cfg.get("num_fold") is not None
    if task_stem is None:
        task_stem = Path(task_yaml_path).stem

    level_pct = f"{level:.4g}"  # display only; for warning messages
    top_prefix = f"{task_stem}@{level_pct}"
    t_top = time.perf_counter()
    _log(top_prefix, f"task_type={task_type} cv={is_cv}; src={src_manifest_dir}")

    # Load the three manifest files. test.json is optional (CV writes only
    # train/valid; non-CV writes all three).
    t_load = time.perf_counter()
    train_src = json.loads((src_manifest_dir / "train.json").read_text())
    valid_src = json.loads((src_manifest_dir / "valid.json").read_text())
    test_path = src_manifest_dir / "test.json"
    test_src = json.loads(test_path.read_text()) if test_path.exists() else None
    _log(top_prefix, f"loaded manifests in {time.perf_counter()-t_load:.2f}s")
    if is_cv:
        # CV path. train_src is list[dict] — one fold per entry. For each
        # fold we subsample only the train portion; the fold's valid serves
        # as held-out test and passes through unchanged.
        new_train = []
        new_valid = []
        for fold_idx, (tr_fold, va_fold) in enumerate(zip(train_src, valid_src)):
            uniq = {_hashable(r["Participant_ID"]) for r in tr_fold.values()}
            # Budget = ceil(unique_speakers × level), floored at 2 so a probe
            # always sees more than one participant.
            n_target = max(2, math.ceil(len(uniq) * level))
            # Seed varies by fold so different folds get different subsamples
            # but the same fold reproduces across reruns.
            rng = np.random.RandomState(_subseed(seed, level, task_stem, fold_idx))
            where = f"{task_stem}@{level_pct}/fold{fold_idx}/train"
            keep = _select_participants(tr_fold, task_type, n_target, rng, where=where)
            new_train.append(_filter_records(tr_fold, keep))
            new_valid.append(va_fold)  # untouched
        t_write = time.perf_counter()
        _write_json(dst_manifest_dir / "train.json", new_train)
        _write_json(dst_manifest_dir / "valid.json", new_valid)
        _log(top_prefix, f"wrote manifests in {time.perf_counter()-t_write:.2f}s")
    else:
        # Non-CV path. train.json and valid.json are independent dict[uid->rec]
        # files. Subsample each separately (the user explicitly clarified this
        # — splits aren't derived from a single combined manifest by `split`,
        # they're independent files). test.json passes through unchanged.
        uniq_tr = {_hashable(r["Participant_ID"]) for r in train_src.values()}
        n_train = max(2, math.ceil(len(uniq_tr) * level))
        rng_tr = np.random.RandomState(_subseed(seed, level, task_stem, "train"))
        keep_tr = _select_participants(
            train_src, task_type, n_train, rng_tr,
            where=f"{task_stem}@{level_pct}/train",
        )
        new_train = _filter_records(train_src, keep_tr)

        uniq_va = {_hashable(r["Participant_ID"]) for r in valid_src.values()}
        n_val = max(2, math.ceil(len(uniq_va) * level))
        rng_va = np.random.RandomState(_subseed(seed, level, task_stem, "valid"))
        keep_va = _select_participants(
            valid_src, task_type, n_val, rng_va,
            where=f"{task_stem}@{level_pct}/valid",
        )
        new_valid = _filter_records(valid_src, keep_va)

        # Match the indent levels written by prep_utils.to_sb_dict_and_save
        # (5 for train/valid, 4 for test) so a diff against the source manifest
        # only surfaces real subsampling deltas.
        t_write = time.perf_counter()
        _write_json(dst_manifest_dir / "train.json", new_train, indent=5)
        _write_json(dst_manifest_dir / "valid.json", new_valid, indent=5)
        if test_src is not None:
            _write_json(dst_manifest_dir / "test.json", test_src, indent=4)
        _log(top_prefix, f"wrote manifests in {time.perf_counter()-t_write:.2f}s")

    _log(top_prefix, f"TOTAL {time.perf_counter()-t_top:.2f}s")

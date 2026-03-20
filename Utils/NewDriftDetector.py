# DriftDetector.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd


@dataclass
class PageHinkleyDriftDetector:
    burn_in_windows: int = 6
    lambda_ph: float = 0.05

    # state
    _cer_history: List[float] = field(default_factory=list)
    _reference_cer: Optional[float] = None
    _sum_m: float = 0.0
    _min_sum_m: float = 0.0

    def update(self, cer: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns:
            perf_drift: bool
            info: dict with internal stats
        """
        info = {"cer": float(cer), "reference_cer": self._reference_cer, "ph_stat": None, "burnin": False}

        # burn-in to set reference
        if self._reference_cer is None:
            self._cer_history.append(float(cer))
            info["burnin"] = True
            if len(self._cer_history) >= self.burn_in_windows:
                self._reference_cer = sum(self._cer_history) / len(self._cer_history)
                self._cer_history.clear()
                self._sum_m = 0.0
                self._min_sum_m = 0.0
            return False, info

        # PH update (no delta, fixed reference)
        dev = float(cer) - float(self._reference_cer)
        self._sum_m += dev
        self._min_sum_m = min(self._min_sum_m, self._sum_m)
        ph_stat = self._sum_m - self._min_sum_m
        info["reference_cer"] = self._reference_cer
        info["ph_stat"] = ph_stat

        perf_drift = ph_stat > self.lambda_ph
        if perf_drift:
            # reset accumulators ONLY (keep reference fixed as requested)
            self._sum_m = 0.0
            self._min_sum_m = 0.0

        return perf_drift, info



@dataclass
class NoveltyBufferManager:

    min_total_unseen_samples: int = 10
    min_unseen_samples_per_class: int = 5
    max_total_unseen_samples: int = 5000
    max_unseen_samples_per_class: int = 1000

    # gating knobs
    min_unseen_ratio_in_window: float = 0.03   # high unseen ratio can trigger training (if sample gates are met)
    max_wait_windows_since_first_novelty: int = 6  # waited long enough -> allow trigger (if sample gates are met)

    # state
    unseen_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    first_novelty_window: Optional[str] = None
    windows_since_first_novelty: int = 0
    per_class_counts: Dict[str, int] = field(default_factory=dict)

    def _cap_append(self, df_new: pd.DataFrame) -> int:
        """
        Append with caps. Very simple policy:
        - enforce per-class cap
        - enforce total cap
        Returns number of rows actually appended.
        """
        if df_new.empty:
            return 0

        # enforce per-class cap
        keep_rows = []
        for idx, row in df_new.iterrows():
            cls = str(row["next_act"])
            curr = self.per_class_counts.get(cls, 0)
            if curr >= self.max_unseen_samples_per_class:
                continue
            keep_rows.append(idx)
            self.per_class_counts[cls] = curr + 1

            # enforce total cap
            if len(self.unseen_df) + len(keep_rows) >= self.max_total_unseen_samples:
                break

        if not keep_rows:
            return 0

        df_keep = df_new.loc[keep_rows].copy()
        self.unseen_df = pd.concat([self.unseen_df, df_keep], ignore_index=True)
        return len(df_keep)

    def add_unseen_samples(self, window_key: str, batch_df: pd.DataFrame, unseen_labels: Set[str]) -> Tuple[int, float, int]:

        n = int(len(batch_df))
        if not unseen_labels or n == 0:
            return 0, 0.0, 0

        if self.first_novelty_window is None:
            self.first_novelty_window = str(window_key)
            self.windows_since_first_novelty = 0

        self.windows_since_first_novelty += 1

        # select only rows where next_act is unseen
        mask = batch_df["next_act"].astype(str).isin(unseen_labels)
        df_unseen = batch_df.loc[mask].copy()
        unseen_cnt = int(len(df_unseen))
        unseen_ratio = float(unseen_cnt / n)

        added = self._cap_append(df_unseen)
        return added, unseen_ratio, unseen_cnt

    def add_novel_samples(self, window_key: str, batch_df: pd.DataFrame, row_mask: List[bool]) -> Tuple[
        int, float, int]:

        n = int(len(batch_df))
        if n == 0:
            return 0, 0.0, 0

        if self.first_novelty_window is None and any(row_mask):
            self.first_novelty_window = str(window_key)
            self.windows_since_first_novelty = 0

        if any(row_mask):
            self.windows_since_first_novelty += 1

        # select rows
        df_novel = batch_df.loc[row_mask].copy()
        novel_cnt = int(len(df_novel))
        novel_ratio = float(novel_cnt / n) if n > 0 else 0.0

        added = self._cap_append(df_novel)
        return added, novel_ratio, novel_cnt

    def sample_gates_satisfied(self) -> bool:
        if len(self.unseen_df) < self.min_total_unseen_samples:
            return False
        return any(cnt >= self.min_unseen_samples_per_class for cnt in self.per_class_counts.values())

    def should_trigger_train(self, novelty: bool, perf_drift: bool, unseen_ratio_in_window: float) -> Tuple[bool, List[str]]:
        """
        AND gating:
          novelty must be True
          AND sample gates (total + per-class) satisfied
          AND (perf_drift OR high_unseen_ratio OR waited_long)
        """
        reasons = []
        if not novelty:
            return False, reasons

        if not self.sample_gates_satisfied():
            return False, reasons

        waited_long = self.windows_since_first_novelty >= self.max_wait_windows_since_first_novelty
        high_ratio = unseen_ratio_in_window >= self.min_unseen_ratio_in_window

        ok = perf_drift or high_ratio or waited_long
        if ok:
            if perf_drift:
                reasons.append("perf_drop")
            if high_ratio:
                reasons.append("high_unseen_ratio")
            if waited_long:
                reasons.append("max_wait_exceeded")

        return ok, reasons

    def clear(self):
        self.unseen_df = pd.DataFrame()
        self.first_novelty_window = None
        self.windows_since_first_novelty = 0
        self.per_class_counts.clear()


@dataclass
class DriftDetector:
    #known_train_labels: Set[str]                      # frozen snapshot from training (next_act label names)
    known_train_events: Set[str]
    ph: PageHinkleyDriftDetector
    buffer: NoveltyBufferManager

    def update(self, window_key: str, batch_df: pd.DataFrame, acc: float) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:

        n = int(len(batch_df))
        cer = 1.0 - float(acc)

        '''
        # ---- novelty tracking (string-level, vs training snapshot) ----
        win_labels = set(batch_df["next_act"].astype(str).tolist())
        unseen_labels = win_labels - self.known_train_labels
        novelty = len(unseen_labels) > 0
        '''

        # unseen labels based on next_act
        next_acts = batch_df["next_act"].astype(str).tolist()
        win_labels = set(next_acts)
        unseen_labels = win_labels - self.known_train_events

        # unseen tokens based on prefix tokens
        unseen_tokens: Set[str] = set()

        # scan prefixes for unseen tokens and mark rows with unseen tokens in prefix
        prefix_has_unseen = []
        for p in batch_df["prefix"].astype(str).tolist():
            toks = p.split()
            has = any(t not in self.known_train_events for t in toks)
            prefix_has_unseen.append(has)
            if has:
                unseen_tokens.update([t for t in toks if t not in self.known_train_events])

        # novelty: either unseen label or unseen token in prefix
        novelty = (len(unseen_labels) > 0) or (len(unseen_tokens) > 0)

        # buffer unseen samples ALWAYS (independent from perf drop)
        #added, unseen_ratio, unseen_cnt = self.buffer.add_unseen_samples(window_key, batch_df, unseen_labels)

        # row is novel if next_act unseen OR prefix has unseen token
        next_act_unseen_mask = batch_df["next_act"].astype(str).isin(unseen_labels).tolist()
        row_mask = [a or b for a, b in zip(next_act_unseen_mask, prefix_has_unseen)]

        added, novel_ratio, novel_cnt = self.buffer.add_novel_samples(window_key, batch_df, row_mask)

        # perf monitoring independent
        perf_drift, perf_info = self.ph.update(cer)

        # AND gating decision
        trigger, trigger_reasons = self.buffer.should_trigger_train(novelty, perf_drift, novel_ratio)

        info = {
            "window_key": str(window_key),
            "n": n,
            "acc": float(acc),
            "cer": float(cer),

            "novelty": novelty,
            "unseen_labels": sorted(list(unseen_labels)),
            "unseen_count_in_window": int(novel_cnt),
            "unseen_ratio_in_window": float(novel_ratio),
            "buffer_added": int(added),
            "buffer_total": int(len(self.buffer.unseen_df)),
            "buffer_per_class_counts": dict(self.buffer.per_class_counts),
            "first_novelty_window": self.buffer.first_novelty_window,
            "windows_since_first_novelty": int(self.buffer.windows_since_first_novelty),

            "perf_drift": bool(perf_drift),
            "perf_info": perf_info,

            "trigger_train": bool(trigger),
            "trigger_reasons": trigger_reasons,
            "sample_gates_ok": bool(self.buffer.sample_gates_satisfied()),
        }

        return trigger, self.buffer.unseen_df, info

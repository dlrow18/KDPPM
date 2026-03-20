import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class PageHinkleyDriftDetector:
    """
    Page-Hinkley Test + 新活动显式检测的混合漂移检测器
    专为固定月度窗口的 next-activity prediction 任务设计

    文献依据：
    - Baier et al. (2020) "Handling Concept Drift for Predictions in Business Process Mining"
      https://arxiv.org/pdf/2005.05810
    - 与 DARWIN (Pasquadibisceglie et al., 2023) 的误差率监控思想高度一致

    使用方法：
    每个窗口预测完成后调用 .update() 即可。
    """

    def __init__(self,
                 lambda_ph: float = 0.6,  # Baier et al. 推荐阈值，可调 0.5~1.0
                 burn_in_windows: int = 8,  # 前 8 个窗口作为稳定参考期（自动计算参考 CER）
                 min_samples_per_window: int = 30,  # 窗口样本太少不触发性能漂移检测
                 verbose: bool = True):

        self.lambda_ph = lambda_ph
        self.burn_in_windows = burn_in_windows
        self.min_samples = min_samples_per_window
        self.verbose = verbose

        # 内部状态
        self.cer_history: List[float] = []
        self.burn_in_cers: List[float] = []
        self.reference_cer: float = 0.0
        self.sum_m: float = 0.0
        self.min_m: float = 0.0
        self.is_in_burn_in: bool = True
        self.drift_count: int = 0

    def update(self, cer: float, win_key: str = None) -> bool:
        """只负责性能漂移检测，返回 bool"""
        if self.is_in_burn_in:
            self.burn_in_cers.append(cer)
            if len(self.burn_in_cers) >= self.burn_in_windows:
                self.reference_cer = np.mean(self.burn_in_cers)
                self.is_in_burn_in = False
                if self.verbose:
                    print(f"[PageH] Burn-in completed. Reference CER = {self.reference_cer:.4f}")
            return False

        deviation = cer - self.reference_cer
        self.sum_m += deviation
        self.min_m = min(self.min_m, self.sum_m)
        if self.sum_m - self.min_m > self.lambda_ph:
            if self.verbose:
                print(f"[PageH] DRIFT detected | stat={self.sum_m - self.min_m:.3f}")
            self.sum_m = self.min_m = 0.0
            return True
        return False

    '''
    def update(self,
               acc: float,
               new_classes_count: int,
               n_samples: int,
               win_key: str = None) -> Tuple[bool, List[str]]:
        """
        更新检测器并返回是否漂移 + 原因列表

        参数：
            acc: 当前窗口准确率 (0~1)
            new_classes_count: 本窗口出现的新活动数量（来自 vocab 扩展）
            n_samples: 当前窗口样本数
            win_key: 窗口标识（如 "2011/10"），仅用于打印

        返回：(is_drift: bool, reasons: list[str])
        """
        cer = 1.0 - acc
        reasons: List[str] = []

        # ---------- 1. 显式新活动检测（强信号，开放集漂移） ----------
        if new_classes_count >= 1:
            reasons.append(f"New activity(ies) detected: +{new_classes_count}")

        # ---------- 2. Page-Hinkley 性能漂移检测 ----------
        self.cer_history.append(cer)

        if self.is_in_burn_in:
            self.burn_in_cers.append(cer)
            if len(self.burn_in_cers) >= self.burn_in_windows:
                self.reference_cer = np.mean(self.burn_in_cers)
                self.is_in_burn_in = False
                if self.verbose:
                    print(f"[DriftDetector] Burn-in completed ({self.burn_in_windows} windows). "
                          f"Reference CER = {self.reference_cer:.4f}")
            return False, reasons  # burn-in 期间不触发漂移

        # 样本量太小跳过性能检测
        if n_samples < self.min_samples:
            return len(reasons) > 0, reasons

        # Page-Hinkley 核心计算
        deviation = cer - self.reference_cer  # 正值表示性能变差
        self.sum_m += deviation
        self.min_m = min(self.min_m, self.sum_m)
        ph_stat = self.sum_m - self.min_m

        if ph_stat > self.lambda_ph:
            reasons.append(f"Page-Hinkley drift detected: stat={ph_stat:.3f} > λ={self.lambda_ph}")
            self._reset_after_drift()
            return True, reasons

        # 仅新活动也算漂移（保守策略）
        return len(reasons) > 0, reasons
    '''

    def _reset_after_drift(self):
        """漂移后部分重置，继续监控后续窗口"""
        self.sum_m = 0.0
        self.min_m = 0.0
        self.drift_count += 1

    def get_stats(self) -> dict:
        """返回检测器统计信息，便于日志和论文实验报告"""
        return {
            "total_windows_processed": len(self.cer_history),
            "burn_in_completed": not self.is_in_burn_in,
            "reference_cer": round(self.reference_cer, 4),
            "drift_count": self.drift_count,
            "current_ph_stat": round(self.sum_m - self.min_m, 4)
        }


class ADWINDriftDetector:
    """
    Batch 适配版 ADWIN（专为月度固定窗口设计）
    文献：DARWIN (2023) + Bifet & Gavalda (2007)
    已针对渐进漂移进行优化
    """

    def __init__(self,
                 delta: float = 0.05,        # 关键！调大后对渐进漂移更敏感（推荐 0.03~0.08）
                 min_window_size: int = 5,   # helpdesk 月窗口少，建议 4~6
                 min_diff: float = 0.08,     # 新增：最小实际差异要求，避免噪声
                 verbose: bool = True):

        self.delta = delta
        self.min_window_size = min_window_size
        self.min_diff = min_diff
        self.verbose = verbose

        self.window: List[float] = []   # 活跃 CER 窗口
        self.drift_count = 0

    def update(self, cer: float, win_key: str = None) -> bool:
        """只负责性能漂移检测，返回 bool"""
        self.window.append(cer)

        if self.verbose:
            print(f"  [ADWIN Debug] {win_key} | CER={cer:.4f} | win_size={len(self.window)}")

        if len(self.window) < self.min_window_size:
            return False

        return self._check_for_drift(win_key)

    '''
    def update(self, acc: float, new_classes_count: int, n_samples: int, win_key: str = None):
        cer = 1.0 - acc
        reasons = []

        if new_classes_count >= 1:
            reasons.append(f"New activity(ies) detected: +{new_classes_count}")

        self.window.append(cer)

        if self.verbose:
            print(f"  [ADWIN Debug] {win_key} | CER={cer:.4f} | win_size={len(self.window)}")

        if len(self.window) < self.min_window_size:
            return len(reasons) > 0, reasons

        drift_detected = self._check_for_drift(win_key)
        if drift_detected:
            reasons.append(f"ADWIN drift detected (delta={self.delta}, min_diff={self.min_diff})")
            self.drift_count += 1

        return len(reasons) > 0, reasons
    '''

    def _check_for_drift(self, win_key=None):
        n = len(self.window)
        # 只检查“最近 30% 窗口 vs 之前”的差异（更适合 batch 渐进漂移）
        cut = max(1, int(n * 0.7))   # 重点检查最近 30% 是否比前面高很多

        left = self.window[:cut]
        right = self.window[cut:]
        n0, n1 = len(left), len(right)

        if n0 < 3 or n1 < 2:
            return False

        mu0 = sum(left) / n0
        mu1 = sum(right) / n1
        diff = abs(mu0 - mu1)

        epsilon = self._get_epsilon(n0, n1)

        if diff > max(self.min_diff, epsilon):
            if self.verbose:
                print(f"[ADWIN] DRIFT DETECTED @ {win_key} | "
                      f"μ_old={mu0:.4f} → μ_new={mu1:.4f} (diff={diff:.4f} > ε={epsilon:.4f})")
            # 丢弃旧窗口
            self.window = right[:]
            return True
        return False

    def _get_epsilon(self, n0, n1):
        n = n0 + n1
        if n == 0: return 0.0
        return math.sqrt((n0 + n1) * math.log(4.0 / self.delta) / (2.0 * n0 * n1))


class DriftBufferManager:
    """只负责收集样本，不做任何决策"""

    def __init__(self, min_samples_for_kd: int = 100,
                 max_confirmation_windows: int = 3,
                 confirmation_acc_drop: float = 0.10,
                 verbose: bool = True):
        self.min_samples = min_samples_for_kd
        self.max_confirm = max_confirmation_windows
        self.confirm_drop = confirmation_acc_drop
        self.verbose = verbose

        self.state = "NORMAL"
        self.buffered_df = pd.DataFrame()
        self.candidate_start_window = 0
        self.current_window_idx = 0
        self.reference_acc = 0.0

    def update(self, win_key: str, batch_df: pd.DataFrame,
               new_classes_count: int, win_acc: float) -> Tuple[bool, Optional[pd.DataFrame], str]:
        self.current_window_idx += 1
        reason = ""

        if new_classes_count >= 1:
            if self.state == "NORMAL":
                self.state = "CANDIDATE"
                self.candidate_start_window = self.current_window_idx
                self.reference_acc = win_acc
                if self.verbose:
                    print(f"[Buffer] ENTER CANDIDATE at {win_key}")

            if new_classes_count > 0:
                self.buffered_df = pd.concat([self.buffered_df, batch_df], ignore_index=True)
                if self.verbose:
                    print(f"  [Buffer] Added {len(batch_df)} samples | Total: {len(self.buffered_df)}")

        if self.state == "CANDIDATE":
            buffered_count = len(self.buffered_df)
            windows_elapsed = self.current_window_idx - self.candidate_start_window + 1
            acc_drop = self.reference_acc - win_acc

            if (buffered_count >= self.min_samples or
                    windows_elapsed >= self.max_confirm or
                    acc_drop >= self.confirm_drop):
                reason = f"Buffered={buffered_count} | Windows={windows_elapsed} | Drop={acc_drop:.3f}"
                trigger_df = self.buffered_df.copy()
                self._reset()
                return True, trigger_df, reason

        return False, None, ""

    def _reset(self):
        self.state = "NORMAL"
        self.buffered_df = pd.DataFrame()


class DriftDetector:
    def __init__(self,
                 detector_type: str = "PageHinkley",  # ← 这里切换！ "PageHinkley" 或 "ADWIN"
                 min_samples_for_kd: int = 100,
                 max_confirmation_windows: int = 3,
                 confirmation_acc_drop: float = 0.10,
                 verbose: bool = True):

        self.verbose = verbose

        if detector_type == "PageHinkley":
            self.perf_detector = PageHinkleyDriftDetector(verbose=verbose)
        else:
            self.perf_detector = ADWINDriftDetector(verbose=verbose)

        self.buffer_manager = DriftBufferManager(
            min_samples_for_kd=min_samples_for_kd,
            max_confirmation_windows=max_confirmation_windows,
            confirmation_acc_drop=confirmation_acc_drop,
            verbose=verbose
        )

    def update(self, win_key: str, batch_df: pd.DataFrame, win_acc: float, new_classes_count: int):
        cer = 1.0 - win_acc
        reasons = []

        if new_classes_count >= 1:
            reasons.append(f"New activity(ies) detected: +{new_classes_count}")

        # 调用性能检测器（现在只需传 cer 和 win_key）
        perf_drift = self.perf_detector.update(cer, win_key)
        if perf_drift:
            reasons.append(f"{type(self.perf_detector).__name__} performance drift")

        # 缓冲 + 触发判定
        should_trigger, buffered_data, buffer_reason = self.buffer_manager.update(
            win_key, batch_df, new_classes_count, win_acc
        )
        if should_trigger:
            reasons.append(f"Buffer trigger: {buffer_reason}")

        return should_trigger or (new_classes_count >= 1 and perf_drift), buffered_data, reasons

    '''
    def update(self, win_key: str, batch_df: pd.DataFrame, win_acc: float, new_classes_count: int):
        cer = 1.0 - win_acc
        reasons = []

        if new_classes_count >= 1:
            reasons.append(f"New activity(ies) detected: +{new_classes_count}")

        # 调用对应检测器的 update 函数
        perf_drift = self.perf_detector.update(cer, win_key)
        if perf_drift:
            reasons.append(f"{type(self.perf_detector).__name__} performance drift")

        # 缓冲 + 触发判定
        should_trigger, buffered_data, buffer_reason = self.buffer_manager.update(
            win_key, batch_df, new_classes_count, win_acc
        )
        if should_trigger:
            reasons.append(f"Buffer trigger: {buffer_reason}")

        return should_trigger or (new_classes_count >= 1 and perf_drift), buffered_data, reasons
    '''
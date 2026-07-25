# PeriodicRetrain/PeriodicRetrain.py
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np

# Allow running from either project root or PeriodicRetrain folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PreProcessing.LogsDataLoader import LogsDataLoader, DynamicVocabManager
from Model.LSTMClassifier import (
    LSTMClassifier,
    train_model,
    predict_model,
    train_model_simulation_style,
    compute_prf1_weighted_sklearn,
)


def save_window_metrics_to_excel(
    records,
    dataset_name: str,
    excel_path: str,
    sheet_name: str = "all_windows",
):
    """
    Save one dataset's window-level records into a shared Excel file.

    Behavior:
    - If the Excel file does not exist, create it.
    - If it exists, preserve rows from other datasets.
    - Replace only rows for the current dataset_name.
    """

    if not records:
        print("[Excel] No records to save.")
        return

    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)

    new_df = pd.DataFrame(records)

    preferred_cols = [
        "dataset",
        "window_index",
        "window_id",
        "n_samples",
        "predicted_count",
        "not_predicted_count",
        "unseen_count",
        "unseen_ratio",
        "unseen_event_ratio",
        "acc",
        "precision",
        "recall",
        "f1",
    ]

    existing_cols = [c for c in preferred_cols if c in new_df.columns]
    other_cols = [c for c in new_df.columns if c not in existing_cols]
    new_df = new_df[existing_cols + other_cols]

    if os.path.exists(excel_path):
        try:
            old_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        except Exception:
            old_df = pd.DataFrame()

        if not old_df.empty and "dataset" in old_df.columns:
            old_df = old_df[old_df["dataset"] != dataset_name]

        all_windows_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_windows_df = new_df.copy()

    sort_cols = [c for c in ["dataset", "window_index"] if c in all_windows_df.columns]
    if sort_cols:
        all_windows_df = all_windows_df.sort_values(sort_cols).reset_index(drop=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        all_windows_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[Excel] Saved dataset '{dataset_name}' to {excel_path} (sheet: {sheet_name})")


def reset_loader_vocab(loader):
    """
    Strict retraining-style reset:
    every retraining builds a fresh vocab/label space from the expanding window.
    """
    loader.vocab_mapper = DynamicVocabManager()


def get_events_from_df(df: pd.DataFrame) -> set:
    """
    Events include both prefix activities and next_act targets.
    This is used for unseen statistics and learned_novel tracking.
    """
    events = set()

    for prefix in df["prefix"].astype(str).tolist():
        events.update(prefix.split())

    events.update(df["next_act"].astype(str).tolist())
    return events


def get_model_known_events(loader) -> set:
    """
    Current model-known event/activity space.

    Since retraining rebuilds both input token vocab and output label vocab,
    we treat their union as the current known event set.
    """
    known = set(loader.vocab_mapper.token_vocab.keys()) | set(loader.vocab_mapper.label_vocab.keys())

    known.discard(loader.vocab_mapper.pad_token)
    known.discard(loader.vocab_mapper.unk_token)

    return known


def get_model_known_input_tokens(loader) -> set:
    """
    Current model-known input token space.
    Used only for deciding whether a prefix can be encoded/predicted.
    """
    known = set(loader.vocab_mapper.token_vocab.keys())

    known.discard(loader.vocab_mapper.pad_token)
    known.discard(loader.vocab_mapper.unk_token)

    return known


def split_predictable_df_like_simulation(batch_df: pd.DataFrame, known_events: set):
    """
    Replicate simulation.py-style prediction filtering.

    Original logic:
        If the observed activity sequence contains an unknown activity,
        no prediction is made for that sample.

    In our prefix-based representation:
        - prefix tokens are the observed activity sequence.
        - If every prefix token is known to the current model, predict.
        - The target next_act is NOT used to decide predictability, because
          in an online next-activity prediction setting it is unknown before prediction.

    Therefore:
        prefix contains unknown activity -> #NP
        prefix fully known -> #P, even if next_act is an unseen target.
    """
    predictable_mask = []
    not_predicted_mask = []

    for _, row in batch_df.iterrows():
        prefix_tokens = str(row["prefix"]).split()
        prefix_known = all(tok in known_events for tok in prefix_tokens)

        predictable_mask.append(prefix_known)
        not_predicted_mask.append(not prefix_known)

    predictable_df = batch_df.loc[predictable_mask].copy()
    not_predicted_df = batch_df.loc[not_predicted_mask].copy()

    return predictable_df, not_predicted_df, predictable_mask, not_predicted_mask


def compute_unseen_event_ratio_like_detector(batch_df, known_events):
    """
    Event-level unseen ratio:
        (# unseen prefix tokens + unseen next_act) /
        (# all prefix tokens + all next_act)
    """
    total_events = 0
    unseen_events = 0

    for prefix in batch_df["prefix"].astype(str).tolist():
        for act in prefix.split():
            total_events += 1
            if act not in known_events:
                unseen_events += 1

    for act in batch_df["next_act"].astype(str).tolist():
        total_events += 1
        if act not in known_events:
            unseen_events += 1

    return unseen_events / total_events if total_events > 0 else 0.0


def compute_unseen_info_like_detector(batch_df, known_events):
    """
    Window-level unseen sample statistics.

    A sample is unseen if:
        - its prefix contains an unseen event, OR
        - its target next_act is unseen.
    """
    n = int(len(batch_df))

    prefix_has_unseen = []
    target_has_unseen = []

    for _, row in batch_df.iterrows():
        prefix_tokens = str(row["prefix"]).split()
        target = str(row["next_act"])

        prefix_has_unseen.append(any(tok not in known_events for tok in prefix_tokens))
        target_has_unseen.append(target not in known_events)

    row_mask = [a or b for a, b in zip(prefix_has_unseen, target_has_unseen)]

    unseen_count = int(sum(row_mask))
    unseen_ratio = unseen_count / n if n > 0 else 0.0

    return {
        "unseen_count_in_window": unseen_count,
        "unseen_ratio_in_window": unseen_ratio,
    }


def subset_metrics(preds, gts, mask):
    """
    Compute acc / weighted precision / recall / f1 on a subset.
    """
    mask = torch.tensor(mask, dtype=torch.bool)

    n = int(mask.sum().item())
    if n == 0:
        return {
            "n": 0,
            "acc": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    sub_preds = preds[mask]
    sub_gts = gts[mask]

    acc = float((sub_preds == sub_gts).sum().item() / n)
    p, r, f1 = compute_prf1_weighted_sklearn(sub_preds, sub_gts)

    return {
        "n": n,
        "acc": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def overall_subset_metrics(pred_list, gt_list):
    """
    Overall metrics for a subset collected across all windows.
    """
    if len(gt_list) == 0:
        return {
            "n": 0,
            "acc": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    preds_tensor = torch.stack(pred_list)
    gts_tensor = torch.stack(gt_list)

    n = len(gt_list)
    acc = float((preds_tensor == gts_tensor).sum().item() / n)
    p, r, f1 = compute_prf1_weighted_sklearn(preds_tensor, gts_tensor)

    return {
        "n": n,
        "acc": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def build_unseen_event_eval_masks(batch_df, current_known_events, learned_novel_events):
    """
    Same style as KDTest.

    current_unseen_target:
        target is unknown before the current window.

    learned_novel_target:
        target was once novel but is now already known by the current model.

    novel_context_old_target:
        prefix contains a learned novel event, while target is old/known.
    """
    next_acts = batch_df["next_act"].astype(str).tolist()
    prefixes = batch_df["prefix"].astype(str).tolist()

    current_unseen_target_mask = [
        y not in current_known_events
        for y in next_acts
    ]

    old_target_mask = [
        y in current_known_events
        for y in next_acts
    ]

    learned_novel_target_mask = [
        (y in learned_novel_events) and (y in current_known_events)
        for y in next_acts
    ]

    prefix_has_learned_novel_mask = []
    for prefix in prefixes:
        toks = prefix.split()
        prefix_has_learned_novel_mask.append(
            any(tok in learned_novel_events for tok in toks)
        )

    novel_context_old_target_mask = [
        prefix_has_novel and old_target
        for prefix_has_novel, old_target in zip(
            prefix_has_learned_novel_mask,
            old_target_mask,
        )
    ]

    return {
        "current_unseen_target": current_unseen_target_mask,
        "learned_novel_target": learned_novel_target_mask,
        "novel_context_old_target": novel_context_old_target_mask,
    }


def train_fresh_model_on_df(args, loader, train_df, device):
    """
    Full retraining from scratch on the expanding window.

    This intentionally does NOT reuse old model weights.
    It rebuilds:
        - token vocabulary
        - label vocabulary
        - input/output dimensions
        - model parameters
    """
    reset_loader_vocab(loader)

    train_loader = loader.encode_and_prepare(
        train_df,
        batch_size=args.batch_size,
        shuffle=True,
        expand_token_vocab=True,
        expand_label_vocab=True,
        unknown_to_unk=False,
        allow_unknown_labels=False,
    )

    vocab_size = len(loader.vocab_mapper.token_vocab)
    num_classes = len(loader.vocab_mapper.label_vocab)
    pad_idx = loader.vocab_mapper.pad_idx

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        padding_idx=pad_idx,
    )

    model, stats = train_model_simulation_style(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        min_delta=args.min_delta,
        device=device,
    )

    return model, stats


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument(
        "--data_dir",
        type=str,
        default="./Data",
        help="root data dir that contains <dataset>/processed/prefixes.csv",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.1,
        help="Fraction used as initial training part.",
    )

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)

    ap.add_argument("--out_dir", type=str, default="runs/ckpts")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--window_type", type=str, default="month", choices=["day", "week", "month"])
    ap.add_argument("--save_excel", action="store_true")
    ap.add_argument(
        "--excel_path",
        type=str,
        default="./PeriodicRetrain/runs/periodic_retrain_window_metrics.xlsx",
    )

    ap.add_argument(
        "--retrain_every",
        type=int,
        default=1,
        help="Retrain every N prediction windows. With monthly windows, 1 means monthly retraining.",
    )

    ap.add_argument("--min_delta", type=float, default=0.01)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    # ===== Load data =====
    loader = LogsDataLoader(
        dataset_name=args.dataset,
        dir_path=args.data_dir,
        window_type=args.window_type,
    )
    loader.load_data()

    initial_train_df, test_df = loader.split_train_test(args.train_ratio)

    observed_train_df = initial_train_df.copy().reset_index(drop=True)

    print(f"[Data] initial_train={len(initial_train_df)} test={len(test_df)}")
    print(f"[Data] max_case_length={loader.max_case_length}")

    # ===== Initial training =====
    print("\n=== Initial training ===")
    model, stats = train_fresh_model_on_df(
        args=args,
        loader=loader,
        train_df=observed_train_df,
        device=device,
    )

    # Current known event space comes from the current model/encoder.
    known_events = get_model_known_events(loader)
    initial_known_events = set(known_events)
    learned_novel_events = set()

    print(f"[Initial known events] {len(known_events)}")

    test_batches = loader.create_batches(test_df)
    print(f"Testing on {len(test_batches)} windows (window_type={args.window_type})")

    all_preds = []
    all_gts = []

    all_learned_novel_target_preds = []
    all_learned_novel_target_gts = []

    all_novel_context_old_target_preds = []
    all_novel_context_old_target_gts = []

    total_predicted_count = 0
    total_not_predicted_count = 0
    retrain_count = 0
    total_correct_predicted = 0
    total_samples_all = 0

    pending_new_events_since_last_retrain = set()

    window_records = []

    for i, (win_key, batch_df) in enumerate(test_batches.items(), start=1):
        print(f"\n=== Predicting window {i}/{len(test_batches)} - {win_key} ===")

        known_events_before_window = set(known_events)
        known_input_tokens_before_window = get_model_known_input_tokens(loader)

        # Evaluation masks over the full window, for unseen target statistics.
        full_eval_masks = build_unseen_event_eval_masks(
            batch_df=batch_df,
            current_known_events=known_events_before_window,
            learned_novel_events=learned_novel_events,
        )

        # Predict every sample in the current window.
        # Unseen prefix activities are mapped to [UNK].
        # Unseen target activities are encoded as unknown labels and therefore
        # cannot be predicted by the current output layer before retraining.
        predicted_count = int(len(batch_df))
        not_predicted_count = 0

        win_loader = loader.encode_and_prepare(
            batch_df,
            batch_size=args.batch_size,
            shuffle=False,
            expand_token_vocab=False,
            expand_label_vocab=False,
            unknown_to_unk=True,
            allow_unknown_labels=True,
        )

        win_acc, win_preds, win_gts = predict_model(
            model,
            win_loader,
            device=device,
        )

        win_p, win_r, win_f1 = compute_prf1_weighted_sklearn(win_preds, win_gts)
        correct_predicted = int((win_preds == win_gts).sum().item())

        acc_predicted_only = (
            correct_predicted / predicted_count
            if predicted_count > 0
            else 0.0
        )

        acc_all_samples = (
            correct_predicted / len(batch_df)
            if len(batch_df) > 0
            else 0.0
        )

        total_predicted_count += predicted_count
        total_not_predicted_count += not_predicted_count

        total_correct_predicted += correct_predicted
        total_samples_all += int(len(batch_df))

        # Subset metrics are computed over all samples because no sample is skipped.
        if predicted_count > 0:
            predicted_eval_df = batch_df.reset_index(drop=True)

            predicted_eval_masks = build_unseen_event_eval_masks(
                batch_df=predicted_eval_df,
                current_known_events=known_events_before_window,
                learned_novel_events=learned_novel_events,
            )

            learned_novel_target_metrics = subset_metrics(
                win_preds,
                win_gts,
                predicted_eval_masks["learned_novel_target"],
            )

            novel_context_old_target_metrics = subset_metrics(
                win_preds,
                win_gts,
                predicted_eval_masks["novel_context_old_target"],
            )

            learned_mask = torch.tensor(
                predicted_eval_masks["learned_novel_target"],
                dtype=torch.bool,
            )

            novel_context_mask = torch.tensor(
                predicted_eval_masks["novel_context_old_target"],
                dtype=torch.bool,
            )

            if learned_mask.sum().item() > 0:
                all_learned_novel_target_preds.extend(list(win_preds[learned_mask]))
                all_learned_novel_target_gts.extend(list(win_gts[learned_mask]))

            if novel_context_mask.sum().item() > 0:
                all_novel_context_old_target_preds.extend(list(win_preds[novel_context_mask]))
                all_novel_context_old_target_gts.extend(list(win_gts[novel_context_mask]))

            all_preds.extend(list(win_preds))
            all_gts.extend(list(win_gts))
        else:
            learned_novel_target_metrics = {
                "n": 0,
                "acc": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            novel_context_old_target_metrics = {
                "n": 0,
                "acc": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }

        current_unseen_target_n = int(sum(full_eval_masks["current_unseen_target"]))
        current_unseen_target_ratio = (
            current_unseen_target_n / len(batch_df)
            if len(batch_df) > 0
            else 0.0
        )

        unseen_info = compute_unseen_info_like_detector(
            batch_df=batch_df,
            known_events=known_events_before_window,
        )

        unseen_event_ratio = compute_unseen_event_ratio_like_detector(
            batch_df=batch_df,
            known_events=known_events_before_window,
        )

        print(
            f"[Window {win_key}] n={len(batch_df)} "
            f"#P={predicted_count} #NP={not_predicted_count} "
            f"correct={correct_predicted} | "
            f"acc_pred_only={acc_predicted_only * 100:.2f}% | "
            f"acc_all={acc_all_samples * 100:.2f}% | "
            f"P={win_p * 100:.2f}% | "
            f"R={win_r * 100:.2f}% | "
            f"F1={win_f1 * 100:.2f}%"
        )

        print(
            f"[Unseen Eval] "
            f"current_unseen_target_n={current_unseen_target_n} "
            f"ratio={current_unseen_target_ratio:.4f} | "
            f"learned_novel_target: n={learned_novel_target_metrics['n']} "
            f"acc={learned_novel_target_metrics['acc']} "
            f"recall={learned_novel_target_metrics['recall']} "
            f"f1={learned_novel_target_metrics['f1']} | "
            f"novel_context_old_target: n={novel_context_old_target_metrics['n']} "
            f"acc={novel_context_old_target_metrics['acc']}"
        )

        window_records.append({
            "dataset": args.dataset,
            "window_index": i,
            "window_id": str(win_key),
            "n_samples": int(len(batch_df)),
            "predicted_count": int(predicted_count),
            "not_predicted_count": int(not_predicted_count),

            "acc": float(win_acc),
            "precision": float(win_p),
            "recall": float(win_r),
            "f1": float(win_f1),

            "unseen_count": int(unseen_info["unseen_count_in_window"]),
            "unseen_ratio": float(unseen_info["unseen_ratio_in_window"]),
            "unseen_event_ratio": float(unseen_event_ratio),

            "current_unseen_target_n": int(current_unseen_target_n),
            "current_unseen_target_ratio": float(current_unseen_target_ratio),

            "learned_novel_target_n": learned_novel_target_metrics["n"],
            "learned_novel_target_acc": learned_novel_target_metrics["acc"],
            "learned_novel_target_precision": learned_novel_target_metrics["precision"],
            "learned_novel_target_recall": learned_novel_target_metrics["recall"],
            "learned_novel_target_f1": learned_novel_target_metrics["f1"],

            "novel_context_old_target_n": novel_context_old_target_metrics["n"],
            "novel_context_old_target_acc": novel_context_old_target_metrics["acc"],
            "novel_context_old_target_precision": novel_context_old_target_metrics["precision"],
            "novel_context_old_target_recall": novel_context_old_target_metrics["recall"],
            "novel_context_old_target_f1": novel_context_old_target_metrics["f1"],

            "retrain_count_before_window": int(retrain_count),
        })

        # ===== Post-window: expand past window =====
        observed_train_df = pd.concat(
            [observed_train_df, batch_df],
            ignore_index=True,
        )

        # Events observed in this window but not known before this window.
        # They become learned only after a retraining actually happens.
        newly_observed_events = get_events_from_df(batch_df) - known_events_before_window
        pending_new_events_since_last_retrain.update(newly_observed_events)

        # ===== Periodic full retraining =====
        should_retrain = (i % args.retrain_every == 0)

        if should_retrain:
            retrain_count += 1

            print(
                f"\n[Periodic Retrain] after window={win_key} | "
                f"retrain_count={retrain_count} | "
                f"training_samples={len(observed_train_df)}"
            )

            model, stats = train_fresh_model_on_df(
                args=args,
                loader=loader,
                train_df=observed_train_df,
                device=device,
            )

            # After retraining, the model-known space changes.
            known_events = get_model_known_events(loader)

            # Mark newly learned novel events.
            learned_after_retrain = pending_new_events_since_last_retrain & known_events
            learned_novel_events.update(learned_after_retrain)
            pending_new_events_since_last_retrain.clear()

            print(
                f"[Periodic Retrain] new known_events={len(known_events)} | "
                f"learned_novel_events={len(learned_novel_events)}"
            )

    # ===== Overall metrics =====
    try:
        if len(all_gts) > 0:
            all_preds_tensor = torch.stack(all_preds)
            all_gts_tensor = torch.stack(all_gts)

            overall_acc_predicted_only = float(
                (all_preds_tensor == all_gts_tensor).sum().item() / len(all_gts_tensor)
            )

            overall_p, overall_r, overall_f1 = compute_prf1_weighted_sklearn(
                all_preds_tensor,
                all_gts_tensor,
            )
        else:
            all_preds_tensor = torch.tensor([], dtype=torch.long)
            all_gts_tensor = torch.tensor([], dtype=torch.long)

            overall_acc_predicted_only = 0.0
            overall_p = 0.0
            overall_r = 0.0
            overall_f1 = 0.0

        overall_acc_all_samples = (
            total_correct_predicted / total_samples_all
            if total_samples_all > 0
            else 0.0
        )

        overall_acc = overall_acc_predicted_only

        print(f"\nOverall accuracy over predicted samples: {overall_acc_predicted_only * 100:.2f}%")
        print(f"Overall accuracy over all samples     : {overall_acc_all_samples * 100:.2f}%")
        print(f"Overall Precision: {overall_p * 100:.2f}%")
        print(f"Overall Recall   : {overall_r * 100:.2f}%")
        print(f"Overall F1       : {overall_f1 * 100:.2f}%")
        print(
            f"Total #P={total_predicted_count} | "
            f"Total #NP={total_not_predicted_count} | "
            f"Total correct={total_correct_predicted} | "
            f"Total samples={total_samples_all}"
        )

        '''
        print(f"\nOverall accuracy over predicted samples: {overall_acc * 100:.2f}%")
        print(f"Overall Precision: {overall_p * 100:.2f}%")
        print(f"Overall Recall   : {overall_r * 100:.2f}%")
        print(f"Overall F1       : {overall_f1 * 100:.2f}%")
        print(f"Total #P={total_predicted_count} | Total #NP={total_not_predicted_count}")
        '''

        overall_learned_novel_target_metrics = overall_subset_metrics(
            all_learned_novel_target_preds,
            all_learned_novel_target_gts,
        )

        overall_novel_context_old_target_metrics = overall_subset_metrics(
            all_novel_context_old_target_preds,
            all_novel_context_old_target_gts,
        )

        window_records.append({
            "dataset": args.dataset,
            "window_index": len(test_batches) + 1,
            "window_id": "overall",
            "n_samples": int(total_predicted_count + total_not_predicted_count),
            "predicted_count": int(total_predicted_count),
            "not_predicted_count": int(total_not_predicted_count),

            "unseen_count": None,
            "unseen_ratio": None,
            "unseen_event_ratio": None,

            "acc": float(overall_acc),
            "precision": float(overall_p),
            "recall": float(overall_r),
            "f1": float(overall_f1),

            "learned_novel_target_n": overall_learned_novel_target_metrics["n"],
            "learned_novel_target_acc": overall_learned_novel_target_metrics["acc"],
            "learned_novel_target_precision": overall_learned_novel_target_metrics["precision"],
            "learned_novel_target_recall": overall_learned_novel_target_metrics["recall"],
            "learned_novel_target_f1": overall_learned_novel_target_metrics["f1"],

            "novel_context_old_target_n": overall_novel_context_old_target_metrics["n"],
            "novel_context_old_target_acc": overall_novel_context_old_target_metrics["acc"],
            "novel_context_old_target_precision": overall_novel_context_old_target_metrics["precision"],
            "novel_context_old_target_recall": overall_novel_context_old_target_metrics["recall"],
            "novel_context_old_target_f1": overall_novel_context_old_target_metrics["f1"],

            "total_retrain_count": int(retrain_count),
        })

    except Exception as e:
        print(f"[Overall] Failed to compute overall metrics: {e}")

    # ===== Save Excel =====
    if args.save_excel:
        save_window_metrics_to_excel(
            records=window_records,
            dataset_name=args.dataset,
            excel_path=args.excel_path,
            sheet_name="all_windows",
        )

    # ===== Save final checkpoint + vocab =====
    ckpt_path = os.path.join(args.out_dir, f"{args.dataset}_periodic_retrain.pt")
    vocab_path = os.path.join(args.out_dir, f"{args.dataset}_periodic_retrain_vocab.json")

    model.save_model(ckpt_path)
    loader.vocab_mapper.save_vocab(vocab_path)

    print(f"[Saved] model={ckpt_path}")
    print(f"[Saved] vocab={vocab_path}")
    print(f"[Done] total_retrain_count={retrain_count}")


if __name__ == "__main__":
    main()
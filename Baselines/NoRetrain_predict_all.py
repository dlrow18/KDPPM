# NoRetrain/NoRetrain.py
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np

# Allow running from either project root or NoRetrain folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PreProcessing.LogsDataLoader import LogsDataLoader
from Model.LSTMClassifier import LSTMClassifier, train_model, predict_model, compute_prf1_weighted_sklearn


def save_window_metrics_to_excel(
    records,
    dataset_name: str,
    excel_path: str,
    sheet_name: str = "all_windows",
):
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


def compute_unseen_event_ratio_like_detector(batch_df, known_events):
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
    n = int(len(batch_df))

    next_acts = batch_df["next_act"].astype(str).tolist()
    win_labels = set(next_acts)
    unseen_labels = win_labels - set(known_events)

    prefix_has_unseen = []
    unseen_tokens = set()

    for prefix in batch_df["prefix"].astype(str).tolist():
        toks = prefix.split()
        has_unseen = any(tok not in known_events for tok in toks)
        prefix_has_unseen.append(has_unseen)

        if has_unseen:
            unseen_tokens.update([tok for tok in toks if tok not in known_events])

    next_act_unseen_mask = batch_df["next_act"].astype(str).isin(unseen_labels).tolist()
    row_mask = [a or b for a, b in zip(next_act_unseen_mask, prefix_has_unseen)]

    unseen_count = int(sum(row_mask))
    unseen_ratio = float(unseen_count / n) if n > 0 else 0.0

    return {
        "unseen_labels": sorted(list(unseen_labels)),
        "unseen_tokens": sorted(list(unseen_tokens)),
        "unseen_count_in_window": unseen_count,
        "unseen_ratio_in_window": unseen_ratio,
    }


def subset_metrics(preds, gts, mask):
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
        help="Fraction used as TRAIN in split_train_test",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)

    ap.add_argument("--out_dir", type=str, default="runs/ckpts")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--window_type", type=str, default="month", choices=["day", "week", "month"])
    ap.add_argument("--save_excel", action="store_true")
    ap.add_argument("--excel_path", type=str, default="runs/no_retrain_window_metrics.xlsx")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    # ===== Load processed prefixes.csv =====
    loader = LogsDataLoader(
        dataset_name=args.dataset,
        dir_path=args.data_dir,
        window_type=args.window_type,
    )
    loader.load_data()

    train_df, test_df = loader.split_train_test(args.train_ratio)

    print(f"[Data] train={len(train_df)} test={len(test_df)}")
    print(f"[Data] max_case_length={loader.max_case_length}")

    # ===== Known events from initial training data only =====
    known_train_events = set()

    for prefix in train_df["prefix"].astype(str).tolist():
        known_train_events.update(prefix.split())

    known_train_events.update(train_df["next_act"].astype(str).tolist())

    print(f"[Known train events] {len(known_train_events)}")

    # ===== Encode training data only: build initial vocab and label space =====
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

    print(f"[Initial model] vocab_size={vocab_size}, num_classes={num_classes}")

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        padding_idx=pad_idx,
    )

    # ===== Train once only =====
    model, stats = train_model(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    # ===== NoRetrain: no learned novel events because model is never updated =====
    learned_novel_events = set()

    test_batches = loader.create_batches(test_df)
    print(f"Testing on {len(test_batches)} windows (window_type={args.window_type})")

    all_preds, all_gts = [], []
    all_accs, all_keys = [], []

    all_learned_novel_target_preds = []
    all_learned_novel_target_gts = []

    all_novel_context_old_target_preds = []
    all_novel_context_old_target_gts = []

    window_records = []

    for i, (win_key, batch_df) in enumerate(test_batches.items(), start=1):
        print(f"\n=== Predicting window {i}/{len(test_batches)} - {win_key} ===")

        known_events_before_window = set(known_train_events)

        eval_masks = build_unseen_event_eval_masks(
            batch_df=batch_df,
            current_known_events=known_events_before_window,
            learned_novel_events=learned_novel_events,
        )

        # Important:
        # NoRetrain predicts every sample in the window.
        # It does NOT expand token vocab or label vocab.
        # Unknown prefix tokens -> [UNK] and are still predicted.
        # Unknown target labels -> -1 and are naturally counted as incorrect.
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

        learned_novel_target_metrics = subset_metrics(
            win_preds,
            win_gts,
            eval_masks["learned_novel_target"],
        )

        novel_context_old_target_metrics = subset_metrics(
            win_preds,
            win_gts,
            eval_masks["novel_context_old_target"],
        )

        learned_mask = torch.tensor(
            eval_masks["learned_novel_target"],
            dtype=torch.bool,
        )

        novel_context_mask = torch.tensor(
            eval_masks["novel_context_old_target"],
            dtype=torch.bool,
        )

        if learned_mask.sum().item() > 0:
            all_learned_novel_target_preds.extend(list(win_preds[learned_mask]))
            all_learned_novel_target_gts.extend(list(win_gts[learned_mask]))

        if novel_context_mask.sum().item() > 0:
            all_novel_context_old_target_preds.extend(list(win_preds[novel_context_mask]))
            all_novel_context_old_target_gts.extend(list(win_gts[novel_context_mask]))

        current_unseen_target_n = int(sum(eval_masks["current_unseen_target"]))
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
            f"acc={win_acc * 100:.2f}% | "
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
        })

        all_accs.append(win_acc)
        all_keys.append(win_key)
        all_preds.extend(list(win_preds))
        all_gts.extend(list(win_gts))

    # ===== Overall metrics =====
    try:
        overall_acc = float((np.array(all_preds) == np.array(all_gts)).mean()) if len(all_gts) else 0.0

        print(f"\nOverall accuracy (micro): {overall_acc * 100:.2f}%")

        all_preds_tensor = torch.stack(all_preds) if len(all_preds) > 0 else torch.tensor([], dtype=torch.long)
        all_gts_tensor = torch.stack(all_gts) if len(all_gts) > 0 else torch.tensor([], dtype=torch.long)

        overall_p, overall_r, overall_f1 = compute_prf1_weighted_sklearn(
            all_preds_tensor,
            all_gts_tensor,
        )

        print(f"Overall Precision: {overall_p * 100:.2f}%")
        print(f"Overall Recall   : {overall_r * 100:.2f}%")
        print(f"Overall F1       : {overall_f1 * 100:.2f}%")

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
            "n_samples": len(all_gts),

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

    # ===== Save checkpoint + vocab =====
    ckpt_path = os.path.join(args.out_dir, f"{args.dataset}_no_retrain.pt")
    vocab_path = os.path.join(args.out_dir, f"{args.dataset}_no_retrain_vocab.json")

    model.save_model(ckpt_path)
    loader.vocab_mapper.save_vocab(vocab_path)

    print(f"[Saved] model={ckpt_path}")
    print(f"[Saved] vocab={vocab_path}")


if __name__ == "__main__":
    main()
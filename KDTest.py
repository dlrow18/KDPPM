import os
import argparse
import torch
import copy
import pandas as pd

from PreProcessing.LogsDataLoader import LogsDataLoader
from Model.LSTMClassifier import LSTMClassifier, train_model, predict_model, compute_prf1_weighted_sklearn
# from Utils.DriftDetector import PageHinkleyDriftDetector
# from Utils.DriftDetector import ADWINDriftDetector
from Utils.NewDriftDetector import PageHinkleyDriftDetector, NoveltyBufferManager, DriftDetector
from Utils.KDPrepare import UNK_TOKEN, prepare_novel_kd_batch, build_teacher_student_models
from Utils.KDTrainer import incremental_kd_update

'''
# functions for recording results to Excel
def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")
'''


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
    - If it exists, preserve all other datasets' rows.
    - Replace only the rows of the current dataset_name.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="e.g., helpdesk")
    ap.add_argument("--data_dir", type=str, default="./data",
                    help="root data dir that contains <dataset>/processed/prefixes.csv")
    ap.add_argument("--train_ratio", type=float, default=0.1,
                    help="Fraction used as TRAIN in split_train_test (CIL2D-style)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="./runs/baseline")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--window_type", type=str, default=None, choices=[None, "day", "week", "month"])
    ap.add_argument("--save_excel", type=bool, default=False)
    ap.add_argument("--excel_path", type=str, default="./runs/window_metrics.xlsx")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    # Load processed prefixes.csv and compute max_case_length
    loader = LogsDataLoader(dataset_name=args.dataset, dir_path=args.data_dir, window_type=args.window_type)
    loader.load_data()  # sorts by last_event_time and sets max_case_length :contentReference[oaicite:8]{index=8}

    # Split train/test (NOTE: in this implementation, train_ratio is the TRAIN fraction) :contentReference[oaicite:9]{index=9}
    train_df, test_df = loader.split_train_test(args.train_ratio)

    # known_train_labels = set(train_df["next_act"].astype(str).tolist())
    known_train_events = set()
    for p in train_df["prefix"].astype(str).tolist():
        known_train_events.update(p.split())

    known_train_events.update(train_df["next_act"].astype(str).tolist())

    # Encode -> DataLoader (inputs: [B,T], labels: one-hot [B,C], lengths: [B]) :contentReference[oaicite:10]{index=10}
    train_loader = loader.encode_and_prepare(train_df, batch_size=args.batch_size, shuffle=True)
    # test_loader = loader.encode_and_prepare(test_df, batch_size=args.batch_size, shuffle=False)

    # Build model using vocab sizes from DynamicVocabManager
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

    ph = PageHinkleyDriftDetector(burn_in_windows=6, lambda_ph=0.05)

    buf = NoveltyBufferManager(
        min_total_unseen_samples=1,
        min_unseen_samples_per_class=1,
        max_total_unseen_samples=5000,
        max_unseen_samples_per_class=2000,
        min_unseen_ratio_in_window=0.001,
        max_wait_windows_since_first_novelty=6
    )

    detector = DriftDetector(
        known_train_events=known_train_events,
        ph=ph,
        buffer=buf
    )

    '''
    drift_detector = DriftDetector(
        detector_type="PageHinkley",  # 改成 "ADWIN" 即可切换
        min_samples_for_kd=100,
        max_confirmation_windows=3,
        confirmation_acc_drop=0.10,
        verbose=True
    )

    drift_detector = ADWINDriftDetector(
        delta=0.05,  # 关键调整
        min_window_size=5, # 前5个月作为稳定参考
        min_diff=0.08,  # 至少 8% 的 CER 差异才触发
        verbose=True
    )

    drift_detector = PageHinkleyDriftDetector(
        lambda_ph=0.6,  # 可根据实验调参
        burn_in_windows=8,  # 前8个月作为稳定参考
        verbose=True
    )
    '''

    # function to compute unseen event ratio in a batch_df given the label_vocab snapshot
    def compute_unseen_event_ratio_like_detector(batch_df, known_events):
        """
        Event-level unseen ratio using the same known-event reference as DriftDetector.

        Definition:
            unseen_event_ratio =
            (# unseen events in prefix tokens + next_act) /
            (# total events in prefix tokens + next_act)

        This is aligned with unseen_ratio, which marks a prefix as unseen if either:
            - its prefix contains unseen tokens, or
            - its next_act is unseen.
        """
        total_events = 0
        unseen_events = 0

        # Count events inside prefixes
        for prefix in batch_df["prefix"].astype(str).tolist():
            for act in prefix.split():
                total_events += 1
                if act not in known_events:
                    unseen_events += 1

        # Count next_act as the target event of each prefix
        for act in batch_df["next_act"].astype(str).tolist():
            total_events += 1
            if act not in known_events:
                unseen_events += 1

        return unseen_events / total_events if total_events > 0 else 0.0

    def subset_metrics(preds, gts, mask):
        """
        Compute acc / weighted P/R/F1 on a subset.
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
        Split window samples into evaluation groups.

        1. current_unseen_target:
           target is not known before this window update.

        2. learned_novel_target:
           target was once novel, but has already been learned.

        3. novel_context_old_target:
           prefix contains a learned novel event, but target is old/known.

        4. old_target:
           target is currently known.
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
        for p in prefixes:
            toks = p.split()
            prefix_has_learned_novel_mask.append(
                any(tok in learned_novel_events for tok in toks)
            )

        novel_context_old_target_mask = [
            prefix_has_novel and old_target
            for prefix_has_novel, old_target in zip(
                prefix_has_learned_novel_mask,
                old_target_mask
            )
        ]

        return {
            "current_unseen_target": current_unseen_target_mask,
            "learned_novel_target": learned_novel_target_mask,
            "novel_context_old_target": novel_context_old_target_mask,
        }

    # Train
    model, stats = train_model(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    old_token_vocab = copy.deepcopy(loader.vocab_mapper.token_vocab)
    old_label_vocab = copy.deepcopy(loader.vocab_mapper.label_vocab)

    learned_novel_events = set()

    # Test by fixed time windows
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


        known_events_before_window = set(detector.known_train_events)

        eval_masks = build_unseen_event_eval_masks(
            batch_df=batch_df,
            current_known_events=known_events_before_window,
            learned_novel_events=learned_novel_events,
        )

        # Encode this window
        win_loader = loader.encode_and_prepare(
            batch_df,
            batch_size=args.batch_size,
            shuffle=False,
            expand_token_vocab=False,
            expand_label_vocab=False,
            unknown_to_unk=True,
            allow_unknown_labels=True,
        )


        #print(f"[Before window] model_vocab={model.vocab_size}, model_classes={model.num_classes}, "
              #f"token_vocab={len(loader.vocab_mapper.token_vocab)}, label_vocab={len(loader.vocab_mapper.label_vocab)}")

        # Predict on this window
        win_acc, win_preds, win_gts = predict_model(model, win_loader, device=device)

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
            dtype=torch.bool
        )

        novel_context_mask = torch.tensor(
            eval_masks["novel_context_old_target"],
            dtype=torch.bool
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


        #print(f"[After window] model_vocab={model.vocab_size}, model_classes={model.num_classes}, "
              #f"token_vocab={len(loader.vocab_mapper.token_vocab)}, label_vocab={len(loader.vocab_mapper.label_vocab)}")

        # print window accuracy + buffer size
        #print(f"[Window {win_key}] n={len(batch_df)}  acc={win_acc * 100:.2f}%")
        print(
            f"[Window {win_key}] n={len(batch_df)} "
            f"acc={win_acc * 100:.2f}% | P={win_p * 100:.2f}% | R={win_r * 100:.2f}% | F1={win_f1 * 100:.2f}%"
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


        is_triggered, unseen_buffer_df, info = detector.update(win_key, batch_df, win_acc)

        unseen_event_ratio = compute_unseen_event_ratio_like_detector(
            batch_df=batch_df,
            known_events=detector.known_train_events,
        )

        print(f"buffer_total={info['buffer_total']}")

        # Record window-level metrics and info for Excel

        window_records.append({
            "dataset": args.dataset,
            "window_index": i,
            "window_id": str(win_key),
            "n_samples": int(len(batch_df)),
            "acc": float(win_acc),
            "precision": float(win_p),
            "recall": float(win_r),
            "f1": float(win_f1),
            "unseen_count": int(info["unseen_count_in_window"]),
            "unseen_ratio": float(info["unseen_ratio_in_window"]),
            "unseen_event_ratio": float(unseen_event_ratio),  # event-level unseen ratio based on old label vocab
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


        # print drift/trigger information when trigger_train=True
        if is_triggered:
            print(
                f"[Drift Detected]"
                #f"\nper_class={info['buffer_per_class_counts']} "
            )

            # ===== Step 1: prepare D_novel for KD =====
            kd_batch = prepare_novel_kd_batch(
                unseen_buffer_df=unseen_buffer_df,
                trigger_window_df=batch_df,
                old_token_vocab=old_token_vocab,
                old_label_vocab=old_label_vocab,
                unk_token=UNK_TOKEN,
            )

            learned_novel_events.update(kd_batch.new_tokens)
            learned_novel_events.update(kd_batch.new_labels)

            #print(f"[KD Prep] D_novel={len(kd_batch.novel_df)}")
            #print(f"[KD Prep] new_tokens={kd_batch.new_tokens}")
            #print(f"[KD Prep] new_labels={kd_batch.new_labels}")

            # expand current vocab/labels for student side only
            loader.vocab_mapper.expand_token_vocab(
                [row.split() for row in kd_batch.novel_df["prefix"].astype(str).tolist()]
            )
            loader.vocab_mapper.expand_label_vocab(
                kd_batch.novel_df["next_act"].astype(str).tolist()
            )

            new_vocab_size = len(loader.vocab_mapper.token_vocab)
            new_num_classes = len(loader.vocab_mapper.label_vocab)

            #print(f"[KD Prep] old_vocab_size={len(old_token_vocab)} -> new_vocab_size={new_vocab_size}")
            #print(f"[KD Prep] old_num_classes={len(old_label_vocab)} -> new_num_classes={new_num_classes}")

            teacher, student = build_teacher_student_models(
                model=model,
                new_vocab_size=new_vocab_size,
                new_num_classes=new_num_classes,
                device=device,
            )

            #print(f"[KD Prep] teacher vocab={teacher.vocab_size}, classes={teacher.num_classes}")
            #print(f"[KD Prep] student vocab={student.vocab_size}, classes={student.num_classes}")

            # ===== Step 2: minimal KD training on D_novel =====
            student, kd_history = incremental_kd_update(
                kd_batch=kd_batch,
                teacher=teacher,
                student=student,
                loader=loader,
                old_token_vocab=old_token_vocab,
                old_label_vocab=old_label_vocab,
                batch_size=32,
                adaptation_epochs=20,
                kd_epochs=4,
                lambda_kd=0.5,
                temperature=2.0,
                adaptation_lr=2e-3,
                kd_lr=1e-3,
                use_kd=True,
                full_finetune_ce_only=True,
                adaptation_val_ratio=0.1,
                adaptation_patience=3,
                adaptation_min_delta=1e-3,
                device=device
            )

            # ===== Step 3: replace deployed model =====
            model = student

            #print(f"[KD Update] new deployed model vocab={model.vocab_size}, classes={model.num_classes}")
            #print(f"[KD Update] snapshot vocab={len(old_token_vocab)}, classes={len(old_label_vocab)}")

            # refresh old space snapshot for the next cycle
            old_token_vocab = copy.deepcopy(loader.vocab_mapper.token_vocab)
            old_label_vocab = copy.deepcopy(loader.vocab_mapper.label_vocab)

            '''
            current_known_events = set(old_token_vocab.keys()) - {
                loader.vocab_mapper.pad_token,
                loader.vocab_mapper.unk_token,
            }

            detector.update_known_events(current_known_events)
            '''

            detector.buffer.clear()

            print("[KD Update] model replaced by student.")



        all_accs.append(win_acc)
        all_keys.append(win_key)
        all_preds.extend(list(win_preds))
        all_gts.extend(list(win_gts))

    # Optional overall accuracy
    try:
        import numpy as np
        overall_acc = float((np.array(all_preds) == np.array(all_gts)).mean()) if len(all_gts) else 0.0
        print(f"\nOverall accuracy (micro): {overall_acc * 100:.2f}%")

        all_preds_tensor = torch.stack(all_preds) if len(all_preds) > 0 else torch.tensor([], dtype=torch.long)
        all_gts_tensor = torch.stack(all_gts) if len(all_gts) > 0 else torch.tensor([], dtype=torch.long)

        overall_p, overall_r, overall_f1 = compute_prf1_weighted_sklearn(all_preds_tensor, all_gts_tensor)

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

        # ==== append overall summary row ====
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

    except Exception:
        pass


    # Save window-level metrics to Excel
    if args.save_excel:
        save_window_metrics_to_excel(
            records=window_records,
            dataset_name=args.dataset,
            excel_path=args.excel_path,
            sheet_name="all_windows",
        )

    # Save checkpoint + vocab
    ckpt_path = os.path.join(args.out_dir, f"{args.dataset}_baseline.pt")
    model.save_model(ckpt_path)
    loader.vocab_mapper.save_vocab(
        os.path.join(args.out_dir, f"{args.dataset}_vocab.json"))
    # print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()

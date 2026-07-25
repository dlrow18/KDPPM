import os
import argparse
import torch
import copy
import time
from PreProcessing.LogsDataLoader import LogsDataLoader
from Model.LSTMClassifier import LSTMClassifier, train_model, predict_model, compute_prf1_weighted_sklearn
from Utils.NewDriftDetector import PageHinkleyDriftDetector, NoveltyBufferManager, DriftDetector
from Utils.OnlineUpdate import perform_incremental_update
from Utils.OnlineEvalMetrics import compute_unseen_event_ratio_like_detector, compute_novel_event_decomposition, subset_metrics, overall_subset_metrics, build_unseen_event_eval_masks, compute_uhs
from Utils.ExperimentIO import save_window_metrics_to_excel, save_checkpoint_and_vocab, build_window_record, build_overall_record


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_optional_metric(value, percentage=False):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%" if percentage else f"{value:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="e.g., helpdesk")
    ap.add_argument("--data_dir", type=str, default="./data",
                    help="root data dir that contains <dataset>/processed/prefixes.csv")
    ap.add_argument("--out_dir", type=str, default="./runs/baseline")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--window_type", type=str, default=None, choices=[None, "day", "week", "month"])

    ap.add_argument("--save_excel", type=bool, default=False)
    ap.add_argument("--excel_path", type=str, default="./runs/window_metrics.xlsx")
    ap.add_argument("--save_checkpoint", action="store_true")

    ap.add_argument("--benchmark", action="store_true",
                    help="Disable logging, metric recording, and file outputs for runtime measurement.")
    #ap.add_argument("--verbose", action="store_true",
    #                help="Print detailed progress logs.")
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                    help="Print detailed progress logs. No print: --no-verbose")

    ap.add_argument("--save_runtime", action=argparse.BooleanOptionalAction, default=True,
        help="Save and print online runtime by default. No save/print: --no-save_runtime"
    )

    ap.add_argument("--uhs_alpha", type=float, default=0.5,
        help="Weight for learned_novel_target_recall in UHS."
    )

    args = ap.parse_args()

    # Fixed experimental configuration
    train_ratio = 0.1
    batch_size = 32
    epochs = 100
    patience = 10
    lr = 0.002
    embedding_dim = 64
    hidden_dim = 128

    if args.benchmark:
        args.verbose = False
        args.save_excel = False
        args.save_checkpoint = False
        record_metrics = False
        compute_diagnostics = False
    else:
        record_metrics = True
        compute_diagnostics = True

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if args.verbose:
        print("Device:", device)

    # Load processed prefixes.csv and compute max_case_length
    loader = LogsDataLoader(dataset_name=args.dataset, dir_path=args.data_dir, window_type=args.window_type)
    loader.load_data()  # sorts by last_event_time and sets max_case_length :contentReference[oaicite:8]{index=8}

    # Split train/test (NOTE: in this implementation, train_ratio is the TRAIN fraction) :contentReference[oaicite:9]{index=9}
    train_df, test_df = loader.split_train_test(train_ratio)

    # known_train_labels = set(train_df["next_act"].astype(str).tolist())
    known_train_events = set()
    for p in train_df["prefix"].astype(str).tolist():
        known_train_events.update(p.split())

    known_train_events.update(train_df["next_act"].astype(str).tolist())

    # Fixed pretraining known-event set.
    # Used only for diagnostic statistics.
    initial_known_events = set(known_train_events)

    # Encode -> DataLoader (inputs: [B,T], labels: one-hot [B,C], lengths: [B]) :contentReference[oaicite:10]{index=10}
    train_loader = loader.encode_and_prepare(train_df, batch_size=batch_size, shuffle=True)
    # test_loader = loader.encode_and_prepare(test_df, batch_size=batch_size, shuffle=False)

    # Build model using vocab sizes from DynamicVocabManager
    vocab_size = len(loader.vocab_mapper.token_vocab)
    num_classes = len(loader.vocab_mapper.label_vocab)
    pad_idx = loader.vocab_mapper.pad_idx

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        padding_idx=pad_idx,
    )

    #ph = PageHinkleyDriftDetector(burn_in_windows=6, lambda_ph=0.05)
    ph = PageHinkleyDriftDetector(burn_in_windows=6, lambda_ph=1e9)

    # parameters for novelty buffer manager
    buf = NoveltyBufferManager(
        min_total_unseen_samples=30,
        min_unseen_samples_per_class=1,
        max_total_unseen_samples=100000,
        min_unseen_ratio_in_window=0.01,
        max_wait_windows_since_first_unseen=50
    )

    detector = DriftDetector(
        known_train_events=known_train_events,
        ph=ph,
        buffer=buf
    )


    # Train
    print("\n[Pretraining] Started.")

    model, stats = train_model(
        model=model,
        dataloader=train_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        device=device,
    )

    print("[Pretraining] Completed.")

    old_token_vocab = copy.deepcopy(loader.vocab_mapper.token_vocab)
    old_label_vocab = copy.deepcopy(loader.vocab_mapper.label_vocab)

    learned_novel_events = set()

    test_batches = loader.create_batches(test_df)

    print("\n[Online] Started.")
    if args.verbose:
        print(
            f"[Online] Processing {len(test_batches)} windows "
            f"(window_type={args.window_type})."
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    online_start_time = time.perf_counter()

    all_preds, all_gts = [], []
    all_accs, all_keys = [], []

    all_learned_novel_target_preds = []
    all_learned_novel_target_gts = []

    all_novel_context_old_target_preds = []
    all_novel_context_old_target_gts = []

    window_records = []

    total_update_count = 0 # for counting how many times the model is updated

    for i, (win_key, batch_df) in enumerate(test_batches.items(), start=1):
        if args.verbose:
            print(f"\n=== Predicting window {i}/{len(test_batches)} | {win_key} ===")

        known_events_before_window = set(detector.known_train_events)

        current_model_known_events = (
                                             set(old_token_vocab.keys())
                                             | set(old_label_vocab.keys())
                                     ) - {
                                         loader.vocab_mapper.pad_token,
                                         loader.vocab_mapper.unk_token,
                                     }

        if compute_diagnostics:
            novel_decomp = compute_novel_event_decomposition(
                batch_df=batch_df,
                initial_known_events=initial_known_events,
                current_model_known_events=current_model_known_events,
            )

            eval_masks = build_unseen_event_eval_masks(
                batch_df=batch_df,
                current_known_events=known_events_before_window,
                learned_novel_events=learned_novel_events,
            )
        else:
            novel_decomp = None
            eval_masks = None

        # Encode this window
        win_loader = loader.encode_and_prepare(
            batch_df,
            batch_size=batch_size,
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

        if compute_diagnostics:
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
        else:
            win_p = win_r = win_f1 = 0.0
            learned_novel_target_metrics = None
            novel_context_old_target_metrics = None
            current_unseen_target_n = 0
            current_unseen_target_ratio = 0.0


        #print(f"[After window] model_vocab={model.vocab_size}, model_classes={model.num_classes}, "
              #f"token_vocab={len(loader.vocab_mapper.token_vocab)}, label_vocab={len(loader.vocab_mapper.label_vocab)}")

        # print window accuracy + buffer size
        #print(f"[Window {win_key}] n={len(batch_df)}  acc={win_acc * 100:.2f}%")
        if args.verbose:
            print(
                f"[Prediction] n={len(batch_df)} "
                f"Acc={win_acc * 100:.2f}% | P={win_p * 100:.2f}% | R={win_r * 100:.2f}% | F1={win_f1 * 100:.2f}%"
            )
        if args.verbose:
            print(
                f"[Unseen Evaluation] "
                f"learned_unseen_target: "
                f"n={learned_novel_target_metrics['n']}, "
                f"acc={format_optional_metric(learned_novel_target_metrics['acc'], True)}, "
                f"recall={format_optional_metric(learned_novel_target_metrics['recall'], True)}, "
                f"f1={format_optional_metric(learned_novel_target_metrics['f1'], True)} | "
                f"learned_unseen_context: "
                f"n={novel_context_old_target_metrics['n']}, "
                f"acc={format_optional_metric(novel_context_old_target_metrics['acc'], True)}"
            )

        is_triggered, unseen_buffer_df, info = detector.update(win_key, batch_df, win_acc)

        if compute_diagnostics:
            unseen_event_ratio = compute_unseen_event_ratio_like_detector(
                batch_df=batch_df,
                known_events=detector.known_train_events,
            )
        else:
            unseen_event_ratio = 0.0

        if args.verbose:
            print(f"[Buffer] total_samples={info['buffer_total']}")

        # Record window-level metrics and info for Excel

        if record_metrics:
            window_records.append(
                build_window_record(
                    dataset_name=args.dataset,
                    window_index=i,
                    window_id=win_key,
                    batch_df=batch_df,
                    win_acc=win_acc,
                    win_p=win_p,
                    win_r=win_r,
                    win_f1=win_f1,
                    info=info,
                    unseen_event_ratio=unseen_event_ratio,
                    current_unseen_target_n=current_unseen_target_n,
                    current_unseen_target_ratio=current_unseen_target_ratio,
                    learned_novel_target_metrics=learned_novel_target_metrics,
                    novel_context_old_target_metrics=novel_context_old_target_metrics,
                    novel_decomp=novel_decomp,
                )
            )

        # print drift/trigger information when trigger_train=True
        if is_triggered:
            total_update_count += 1

            if args.verbose:
                trigger_reasons = ", ".join(info["trigger_reasons"])
                print(f"[Trigger] Activated | reasons={trigger_reasons}")

            (
                model,
                old_token_vocab,
                old_label_vocab,
                learned_novel_events,
                kd_history,
            ) = perform_incremental_update(
                model=model,
                loader=loader,
                batch_df=batch_df,
                unseen_buffer_df=unseen_buffer_df,
                old_token_vocab=old_token_vocab,
                old_label_vocab=old_label_vocab,
                learned_novel_events=learned_novel_events,
                device=device,
                use_kd=True,
                verbose=args.verbose,
            )

            detector.buffer.clear()

        all_accs.append(win_acc)
        all_keys.append(win_key)
        all_preds.extend(list(win_preds))
        all_gts.extend(list(win_gts))

        if device.type == "cuda":
            torch.cuda.synchronize()
        online_runtime = time.perf_counter() - online_start_time

    # Optional overall accuracy
    try:
        import numpy as np
        overall_acc = float((np.array(all_preds) == np.array(all_gts)).mean()) if len(all_gts) else 0.0
        if args.verbose:
            print(f"\nOverall accuracy (micro): {overall_acc * 100:.2f}%")

        all_preds_tensor = torch.stack(all_preds) if len(all_preds) > 0 else torch.tensor([], dtype=torch.long)
        all_gts_tensor = torch.stack(all_gts) if len(all_gts) > 0 else torch.tensor([], dtype=torch.long)

        overall_p, overall_r, overall_f1 = compute_prf1_weighted_sklearn(all_preds_tensor, all_gts_tensor)

        if args.verbose:
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

        # Calculate UHS
        learned_novel_target_recall = (
            overall_learned_novel_target_metrics["recall"]
        )
        novel_context_old_target_acc = (
            overall_novel_context_old_target_metrics["acc"]
        )

        overall_uhs = compute_uhs(
            learned_novel_target_recall=learned_novel_target_recall,
            novel_context_old_target_acc=novel_context_old_target_acc,
            alpha=args.uhs_alpha,
        )

        if args.verbose:
            print(
                f"Overall UHS     : "
                f"{overall_uhs * 100:.2f}%" if overall_uhs is not None else "Overall UHS     : None"
            )

        # ==== append overall summary row ====
        if record_metrics:
            window_records.append(
                build_overall_record(
                    dataset_name=args.dataset,
                    window_index=len(test_batches) + 1,
                    n_samples=len(all_gts),
                    total_update_count=total_update_count,
                    overall_acc=overall_acc,
                    overall_p=overall_p,
                    overall_r=overall_r,
                    overall_f1=overall_f1,
                    overall_uhs=overall_uhs,
                    overall_learned_novel_target_metrics=overall_learned_novel_target_metrics,
                    overall_novel_context_old_target_metrics=overall_novel_context_old_target_metrics,
                )
            )

    except Exception as e:
        print(f"[Overall Metrics] skipped due to error: {e}")

    if args.save_runtime:
        os.makedirs(args.out_dir, exist_ok=True)
        runtime_path = os.path.join(args.out_dir, f"{args.dataset}_online_runtime.txt")

        with open(runtime_path, "w", encoding="utf-8") as f:
            f.write(f"dataset={args.dataset}\n")
            f.write(f"window_type={args.window_type}\n")
            f.write(f"n_windows={len(test_batches)}\n")
            f.write(f"total_update_count={total_update_count}\n")
            f.write(f"online_runtime_seconds={online_runtime:.6f}\n")
            f.write(f"online_runtime_hms={format_seconds(online_runtime)}\n")


    # Save window-level metrics to Excel
    if args.save_excel and record_metrics:
        save_window_metrics_to_excel(
            records=window_records,
            dataset_name=args.dataset,
            excel_path=args.excel_path,
            sheet_name="all_windows",
        )

    # Save checkpoint + vocab
    if args.save_checkpoint:
        save_checkpoint_and_vocab(
            model=model,
            loader=loader,
            out_dir=args.out_dir,
            dataset=args.dataset,
        )

    print(
        f"[Run] Completed | "
        f"dataset={args.dataset} | "
        f"windows={len(test_batches)} | "
        f"updates={total_update_count} | "
        f"online_runtime={format_seconds(online_runtime)}."
    )

if __name__ == "__main__":
    main()

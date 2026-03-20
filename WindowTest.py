import os
import argparse
import torch

from PreProcessing.LogsDataLoader import LogsDataLoader
from Model.LSTMClassifier import LSTMClassifier, train_model, predict_model
# from Utils.DriftDetector import PageHinkleyDriftDetector
# from Utils.DriftDetector import ADWINDriftDetector
from Utils.DriftDetector import DriftDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="e.g., helpdesk")
    ap.add_argument("--data_dir", type=str, default="./data", help="root data dir that contains <dataset>/processed/prefixes.csv")
    ap.add_argument("--train_ratio", type=float, default=0.1, help="Fraction used as TRAIN in split_train_test (CIL2D-style)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="./runs/baseline")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--window_type", type=str, default=None, choices=[None, "day", "week", "month"], help="Split TEST into fixed time windows (CIL2D-style)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    # 1) Load processed prefixes.csv and compute max_case_length
    loader = LogsDataLoader(dataset_name=args.dataset, dir_path=args.data_dir, window_type=args.window_type)
    loader.load_data()  # sorts by last_event_time and sets max_case_length :contentReference[oaicite:8]{index=8}

    # 2) Split train/test (NOTE: in this implementation, train_ratio is the TRAIN fraction) :contentReference[oaicite:9]{index=9}
    train_df, test_df = loader.split_train_test(args.train_ratio)

    # 3) Encode -> DataLoader (inputs: [B,T], labels: one-hot [B,C], lengths: [B]) :contentReference[oaicite:10]{index=10}
    train_loader = loader.encode_and_prepare(train_df, batch_size=args.batch_size, shuffle=True)
    #test_loader = loader.encode_and_prepare(test_df, batch_size=args.batch_size, shuffle=False)

    # 4) Build model using vocab sizes from DynamicVocabManager
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

    drift_detector = DriftDetector(
        detector_type="PageHinkley",  # 改成 "ADWIN" 即可切换
        min_samples_for_kd=100,
        max_confirmation_windows=3,
        confirmation_acc_drop=0.10,
        verbose=True
    )
    '''
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


    # 5) Train + early stopping
    model, stats = train_model(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    # 6) Test by fixed time windows (CIL2D-style)
    test_batches = loader.create_batches(test_df)  # {win_key: batch_df} :contentReference[oaicite:4]{index=4}
    print(f"Testing on {len(test_batches)} windows (window_type={args.window_type})")

    all_preds, all_gts = [], []
    all_accs, all_keys = [], []

    for i, (win_key, batch_df) in enumerate(test_batches.items(), start=1):
        print(f"\n=== Predicting window {i}/{len(test_batches)} - {win_key} ===")

        # Encode this window (NOTE: this may expand token_vocab/label_vocab) :contentReference[oaicite:5]{index=5}
        win_loader = loader.encode_and_prepare(batch_df, batch_size=args.batch_size, shuffle=False)

        current_num_classes = model.num_classes  # ← 新增
        new_num_classes = len(loader.vocab_mapper.label_vocab)  # ← 新增
        new_classes_count = new_num_classes - current_num_classes  # ← 新增（必须在 expand 之前！）

        # --- Strategy 2: immediately expand model structure if vocab/labels grew (CIL2D-style) :contentReference[oaicite:6]{index=6}
        new_vocab_size = len(loader.vocab_mapper.token_vocab)
        new_num_classes = len(loader.vocab_mapper.label_vocab)

        if new_vocab_size > model.vocab_size:
            model.expand_vocab(new_vocab_size)  # you will add this method in LSTMClassifier.py

        if new_num_classes > model.num_classes:
            model.expand_num_classes(new_num_classes)  # you will add this method in LSTMClassifier.py

        # Predict on this window
        win_acc, win_preds, win_gts = predict_model(model, win_loader, device=device)

        print(f"[Window {win_key}] n={len(batch_df)}  acc={win_acc * 100:.2f}%")

        # Concept drift detection
        #new_num_classes = len(loader.vocab_mapper.label_vocab)
        #new_classes_count = new_num_classes - model.num_classes

        is_trigger, buffered_data, reasons = drift_detector.update(
            win_key=win_key,
            batch_df=batch_df,
            win_acc=win_acc,
            new_classes_count=new_classes_count
        )

        '''
        # ==================== 安全调试输出（重点看这部分） ====================
        has_new_class = new_classes_count >= 1
        has_perf_drift = any("performance drift" in r.lower() for r in reasons)
        has_buffer_trigger = any("Buffer trigger" in r for r in reasons)

        print(f"  [DEBUG @ {win_key}] "
              f"new_classes_count = {new_classes_count:2d}  | "
              f"perf_drift = {has_perf_drift}  | "
              f"buffer_trigger = {has_buffer_trigger}  | "
              f"final_trigger = {is_trigger}")

        if has_new_class:
            print(f"    → 有新类出现！数量 = {new_classes_count}")

        if not has_new_class and has_perf_drift:
            print(f"    → 【重要】有性能漂移，但没有新类 → 因此未触发KD")
        # =====================================================================
        '''

        if is_trigger and buffered_data is not None:
            print(f"\n[KD TRIGGERED @ {win_key}] Reasons: {reasons} ")

        if hasattr(drift_detector, 'buffer_manager') and drift_detector.buffer_manager.state == "CANDIDATE":
            df = drift_detector.buffer_manager.buffered_df
            if not df.empty:
                class_counts = df['next_act'].value_counts()
                print(f"  [Buffer Debug @ {win_key}] "
                      f"Buffered new classes: {len(class_counts)} | "
                      f"Total samples: {len(df)}")
                for act, cnt in class_counts.items():
                    print(f"    → {act}: {cnt} samples")

        '''
        is_drift, reasons = drift_detector.update(
            acc=win_acc,
            new_classes_count=new_classes_count,
            n_samples=len(batch_df),
            win_key=win_key
        )

        if is_drift:
            print(f"\n[DRIFT DETECTED @ {win_key}] Reasons: {reasons}")
            print(f"   Current acc = {win_acc * 100:.2f}%, n={len(batch_df)}")
        '''

        all_accs.append(win_acc)
        all_keys.append(win_key)
        all_preds.extend(list(win_preds))
        all_gts.extend(list(win_gts))

    # Optional overall (micro) accuracy
    try:
        import numpy as np
        overall_acc = float((np.array(all_preds) == np.array(all_gts)).mean()) if len(all_gts) else 0.0
        print(f"\nOVERALL accuracy (micro): {overall_acc * 100:.2f}%")
    except Exception:
        pass

    # 7) Save checkpoint + vocab
    ckpt_path = os.path.join(args.out_dir, f"{args.dataset}_baseline.pt")
    model.save_model(ckpt_path)
    loader.vocab_mapper.save_vocab(os.path.join(args.out_dir, f"{args.dataset}_vocab.json"))  # optional :contentReference[oaicite:11]{index=11}
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()

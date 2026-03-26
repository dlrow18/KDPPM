import os
import argparse
import torch
import copy

from PreProcessing.LogsDataLoader import LogsDataLoader
from Model.LSTMClassifier import LSTMClassifier, train_model, predict_model
# from Utils.DriftDetector import PageHinkleyDriftDetector
# from Utils.DriftDetector import ADWINDriftDetector
from Utils.NewDriftDetector import PageHinkleyDriftDetector, NoveltyBufferManager, DriftDetector
from Utils.KDPrepare import UNK_TOKEN, prepare_novel_kd_batch, build_teacher_student_models
from Utils.KDTrainer import incremental_kd_update


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
    ap.add_argument("--window_type", type=str, default=None, choices=[None, "day", "week", "month"],
                    help="Split TEST into fixed time windows (CIL2D-style)")
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
        min_total_unseen_samples=30,
        min_unseen_samples_per_class=2,
        max_total_unseen_samples=5000,
        max_unseen_samples_per_class=1000,
        min_unseen_ratio_in_window=0.03,
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


    # Test by fixed time windows
    test_batches = loader.create_batches(test_df)
    print(f"Testing on {len(test_batches)} windows (window_type={args.window_type})")

    all_preds, all_gts = [], []
    all_accs, all_keys = [], []

    for i, (win_key, batch_df) in enumerate(test_batches.items(), start=1):
        print(f"\n=== Predicting window {i}/{len(test_batches)} - {win_key} ===")

        # Encode this window
        '''
        win_loader = loader.encode_and_prepare(batch_df, batch_size=args.batch_size, shuffle=False)

        current_num_classes = model.num_classes
        new_num_classes = len(loader.vocab_mapper.label_vocab)
        new_classes_count = new_num_classes - current_num_classes

        new_vocab_size = len(loader.vocab_mapper.token_vocab)
        new_num_classes = len(loader.vocab_mapper.label_vocab)

        if new_vocab_size > model.vocab_size:
            model.expand_vocab(new_vocab_size)

        if new_num_classes > model.num_classes:
            model.expand_num_classes(new_num_classes)
        '''
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


        #print(f"[After window] model_vocab={model.vocab_size}, model_classes={model.num_classes}, "
              #f"token_vocab={len(loader.vocab_mapper.token_vocab)}, label_vocab={len(loader.vocab_mapper.label_vocab)}")

        # print window accuracy + buffer size
        print(f"[Window {win_key}] n={len(batch_df)}  acc={win_acc * 100:.2f}%")

        is_triggered, unseen_buffer_df, info = detector.update(win_key, batch_df, win_acc)

        print(f"buffer_total={info['buffer_total']}")



        # print drift/trigger information when trigger_train=True
        if is_triggered:
            print(
                f"[Drift Detected]"
                #f"\nper_class={info['buffer_per_class_counts']} "
            )

            # ===== Step 1: prepare D_novel for KD =====
            kd_batch = prepare_novel_kd_batch(
                unseen_buffer_df=unseen_buffer_df,
                old_token_vocab=old_token_vocab,
                old_label_vocab=old_label_vocab,
                unk_token=UNK_TOKEN,
            )

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
                adaptation_lr=5e-3,
                kd_lr=1e-3,
                use_kd=True,
                adaptation_val_ratio=0.2,
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
    except Exception:
        pass

    # Save checkpoint + vocab
    ckpt_path = os.path.join(args.out_dir, f"{args.dataset}_baseline.pt")
    model.save_model(ckpt_path)
    loader.vocab_mapper.save_vocab(
        os.path.join(args.out_dir, f"{args.dataset}_vocab.json"))
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()

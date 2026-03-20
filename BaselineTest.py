import os
import argparse
import torch

from PreProcessing.LogsDataLoader import LogsDataLoader
from Model.LSTMClassifier import LSTMClassifier, train_model, predict_model


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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Device:", device)

    # 1) Load processed prefixes.csv and compute max_case_length
    loader = LogsDataLoader(dataset_name=args.dataset, dir_path=args.data_dir, window_type=None)
    loader.load_data()  # sorts by last_event_time and sets max_case_length :contentReference[oaicite:8]{index=8}

    # 2) Split train/test (NOTE: in this implementation, train_ratio is the TRAIN fraction) :contentReference[oaicite:9]{index=9}
    train_df, test_df = loader.split_train_test(args.train_ratio)

    # 3) Encode -> DataLoader (inputs: [B,T], labels: one-hot [B,C], lengths: [B]) :contentReference[oaicite:10]{index=10}
    train_loader = loader.encode_and_prepare(train_df, batch_size=args.batch_size, shuffle=True)
    test_loader = loader.encode_and_prepare(test_df, batch_size=args.batch_size, shuffle=False)

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

    # 5) Train + early stopping
    model, stats = train_model(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    # 6) Test
    acc, preds, gts = predict_model(model, test_loader, device=device)
    print(f"TEST accuracy: {acc*100:.2f}%")

    # 7) Save checkpoint + vocab
    ckpt_path = os.path.join(args.out_dir, f"{args.dataset}_baseline.pt")
    model.save_model(ckpt_path)
    loader.vocab_mapper.save_vocab(os.path.join(args.out_dir, f"{args.dataset}_vocab.json"))  # optional :contentReference[oaicite:11]{index=11}
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()

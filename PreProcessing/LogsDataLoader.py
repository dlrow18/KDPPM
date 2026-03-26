import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
class DynamicVocabManager:
    def __init__(self, pad_token="[PAD]"):
        # Input token vocabulary
        self.token_vocab = {pad_token: 0}
        self.pad_idx = 0
        # Output label vocabulary
        self.label_vocab = {}
        self.pad_token = pad_token
'''

class DynamicVocabManager:
    def __init__(self, pad_token="[PAD]", unk_token="[UNK]"):
        # Input token vocabulary
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.token_vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
        }
        self.pad_idx = self.token_vocab[self.pad_token]
        self.unk_idx = self.token_vocab[self.unk_token]

        # Output label vocabulary
        self.label_vocab = {}



    def expand_token_vocab(self, token_lists):
        """Add new tokens to the vocabulary if they don't exist"""
        new_tokens = set()
        for seq in token_lists:
            for token in seq:
                if token not in self.token_vocab:
                    new_tokens.add(token)

        # Add all new tokens at once to avoid race conditions
        for token in new_tokens:
            self.token_vocab[token] = len(self.token_vocab)

        return len(new_tokens) > 0  # Return whether vocab was expanded

    def expand_label_vocab(self, labels):
        """Add new labels to the vocabulary if they don't exist"""
        new_labels = set()
        for label in labels:
            if label not in self.label_vocab:
                new_labels.add(label)

        # Add all new labels at once
        for label in new_labels:
            self.label_vocab[label] = len(self.label_vocab)

        return len(new_labels) > 0  # Return whether vocab was expanded

    '''
    def encode_inputs(self, token_lists, max_seq_len):
        """
        Args:
            token_lists: List[List[str]]
            max_seq_len: int
        Returns:
            input_ids: LongTensor [B, T]
            lengths: LongTensor [B]
        """
        # Expand vocab
        self.expand_token_vocab(token_lists)
        batch_ids = []
        lengths = []
        for seq in token_lists:
            idxs = [self.token_vocab.get(tok) for tok in seq]
            seq_len = min(len(idxs), max_seq_len)
            lengths.append(seq_len)
            # truncate head if too long
            if len(idxs) > max_seq_len:
                idxs = idxs[-max_seq_len:]
            # pad the sequence if too short
            pad_len = max_seq_len - len(idxs)
            batch_ids.append([self.pad_idx] * pad_len + idxs)
        return torch.tensor(batch_ids, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
    '''

    def encode_inputs(self, token_lists, max_seq_len, expand_vocab=True, unknown_to_unk=False):
        """
        Args:
            token_lists: List[List[str]]
            max_seq_len: int
            expand_vocab: whether to expand token_vocab with unseen tokens
            unknown_to_unk: if True, unseen tokens map to UNK instead of expanding
        Returns:
            input_ids: LongTensor [B, T]
            lengths: LongTensor [B]
        """
        if expand_vocab:
            self.expand_token_vocab(token_lists)

        batch_ids = []
        lengths = []

        for seq in token_lists:
            idxs = []
            for tok in seq:
                if tok in self.token_vocab:
                    idxs.append(self.token_vocab[tok])
                else:
                    if unknown_to_unk:
                        idxs.append(self.unk_idx)
                    else:
                        raise KeyError(f"Unknown token '{tok}' encountered while expand_vocab=False")

            seq_len = min(len(idxs), max_seq_len)
            lengths.append(seq_len)

            if len(idxs) > max_seq_len:
                idxs = idxs[-max_seq_len:]

            pad_len = max_seq_len - len(idxs)
            batch_ids.append([self.pad_idx] * pad_len + idxs)

        return torch.tensor(batch_ids, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


    '''
    def encode_labels(self, label_list):
        """
        Args:
            label_list: List[str]
        Returns:
            labels: LongTensor [B]
        """
        self.expand_label_vocab(label_list)
        idxs = [self.label_vocab.get(lbl) for lbl in label_list]
        # one-hot encode labels for data augmentation in feature space
        one_hot_encoded_labels = np.eye(len(self.label_vocab))[idxs]
        return torch.tensor(one_hot_encoded_labels, dtype=torch.long)
    '''

    def encode_labels(self, label_list, expand_vocab=True, allow_unknown=False):
        """
        Args:
            expand_vocab: whether to expand label_vocab with unseen labels
            allow_unknown: if True, unseen labels are encoded as -1 (for inference-only use)
        Returns:
            labels: LongTensor [B, C] if all labels known
                    or LongTensor [B] with -1 for unknown labels when allow_unknown=True
        """
        if expand_vocab:
            self.expand_label_vocab(label_list)

        idxs = []
        has_unknown = False
        for lbl in label_list:
            if lbl in self.label_vocab:
                idxs.append(self.label_vocab[lbl])
            else:
                if allow_unknown:
                    idxs.append(-1)
                    has_unknown = True
                else:
                    raise KeyError(f"Unknown label '{lbl}' encountered while expand_vocab=False")

        if has_unknown:
            return torch.tensor(idxs, dtype=torch.long)

        one_hot_encoded_labels = np.eye(len(self.label_vocab))[idxs]
        return torch.tensor(one_hot_encoded_labels, dtype=torch.long)



    def save_vocab(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'token_vocab': self.token_vocab, 'label_vocab': self.label_vocab}, f)

    '''
    def load_vocab(self, path):
        with open(path) as f:
            data = json.load(f)
        self.token_vocab = data['token_vocab']
        self.label_vocab = data['label_vocab']
        self.pad_idx = self.token_vocab.get(self.pad_token, 0)
    '''

    def load_vocab(self, path):
        with open(path) as f:
            data = json.load(f)

        self.token_vocab = data['token_vocab']
        self.label_vocab = data['label_vocab']

        if self.unk_token not in self.token_vocab:
            self.token_vocab[self.unk_token] = len(self.token_vocab)

        self.pad_idx = self.token_vocab.get(self.pad_token, 0)
        self.unk_idx = self.token_vocab[self.unk_token]


class LogsDataLoader:
    def __init__(self, dataset_name, dir_path="./data", window_type=None):
        self.dataset_name = dataset_name
        self.dir_path = f"{dir_path}/{dataset_name}/processed"
        self.window_type = window_type

        self.traces = None
        self.max_case_length = 0
        self.train_df = None
        self.test_df = None
        self.vocab_mapper = DynamicVocabManager()

    def load_data(self):
        prefixes_file = f"{self.dir_path}/prefixes.csv"
        if not os.path.exists(prefixes_file):
            raise FileNotFoundError(f"Prefixes file not found at {prefixes_file}")
        self.traces = pd.read_csv(prefixes_file)
        self.traces['last_event_time'] = pd.to_datetime(self.traces['last_event_time'])
        self.traces = self.traces.sort_values('last_event_time').reset_index(drop=True)
        raw_max = max(self.traces['prefix'].apply(lambda x: len(x.split())))
        self.max_case_length = min(raw_max, 40)
        # print(f"Loaded {len(self.traces)} traces; max_case_length={self.max_case_length}")
        return self.traces

    def split_train_test(self, train_test_ratio):
        if self.traces is None:
            self.load_data()
        sorted_traces = self.traces.sort_values('last_event_time').reset_index(drop=True)
        split_idx = int(len(sorted_traces) * train_test_ratio)
        self.train_df = sorted_traces.iloc[:split_idx].copy()
        self.test_df = sorted_traces.iloc[split_idx:].copy()
        # print(f"Split at {self.train_df['last_event_time'].max()} | train={len(self.train_df)} | test={len(self.test_df)}")
        return self.train_df, self.test_df

    def create_batches(self, df):
        if self.window_type is None:
            return {'full': df}
        key_map = {'day': '%Y-%m-%d', 'month': '%Y/%m'}
        if self.window_type in ('day', 'month'):
            col = self.window_type
            df[col] = df['last_event_time'].dt.strftime(key_map[self.window_type])
        elif self.window_type == 'week':
            df['week'] = df['last_event_time'].apply(lambda x: f"{x.year}/{x.isocalendar()[1]:02d}")
            col = 'week'
        else:
            raise ValueError(f"Invalid window_type: {self.window_type}")
        batches = {grp: grp_df.drop(col, axis=1) for grp, grp_df in df.groupby(col)}
        # print(f"Created {len(batches)} batches by {self.window_type}")
        return batches

    '''
    def encode_and_prepare(self, df, batch_size=32, shuffle=True):
        # Get prefixes and next_act
        token_seqs = [row.split() for row in df['prefix'].values]
        labels = df['next_act'].tolist()
        # Encode inputs / labels and expand vocab if unseen
        input_tensor, lengths = self.vocab_mapper.encode_inputs(token_seqs, self.max_case_length)
        label_tensor = self.vocab_mapper.encode_labels(labels)
        # Prepare DataLoader
        dataset = torch.utils.data.TensorDataset(input_tensor, label_tensor, lengths)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return loader
    '''

    def encode_and_prepare(
            self,
            df,
            batch_size=32,
            shuffle=True,
            expand_token_vocab=True,
            expand_label_vocab=True,
            unknown_to_unk=False,
            allow_unknown_labels=False,
    ):
        token_seqs = [row.split() for row in df['prefix'].values]
        labels = df['next_act'].tolist()

        input_tensor, lengths = self.vocab_mapper.encode_inputs(
            token_seqs,
            self.max_case_length,
            expand_vocab=expand_token_vocab,
            unknown_to_unk=unknown_to_unk,
        )

        label_tensor = self.vocab_mapper.encode_labels(
            labels,
            expand_vocab=expand_label_vocab,
            allow_unknown=allow_unknown_labels,
        )

        dataset = torch.utils.data.TensorDataset(input_tensor, label_tensor, lengths)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        return loader

    def save_metadata(self):
        meta = {
            'max_case_length': self.max_case_length,
            'token_vocab': self.vocab_mapper.token_vocab,
            'label_vocab': self.vocab_mapper.label_vocab
        }
        os.makedirs(self.dir_path, exist_ok=True)
        with open(f"{self.dir_path}/metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)

    '''
    def load_metadata(self):
        with open(f"{self.dir_path}/metadata.json") as f:
            meta = json.load(f)
        self.max_case_length = meta['max_case_length']
        self.vocab_mapper.token_vocab = meta['token_vocab']
        self.vocab_mapper.label_vocab = meta['label_vocab']
        self.vocab_mapper.pad_idx = self.vocab_mapper.token_vocab.get(self.vocab_mapper.pad_token)
        '''

    def load_metadata(self):
        with open(f"{self.dir_path}/metadata.json") as f:
            meta = json.load(f)

        self.max_case_length = meta['max_case_length']
        self.vocab_mapper.token_vocab = meta['token_vocab']
        self.vocab_mapper.label_vocab = meta['label_vocab']

        if self.vocab_mapper.unk_token not in self.vocab_mapper.token_vocab:
            self.vocab_mapper.token_vocab[self.vocab_mapper.unk_token] = len(self.vocab_mapper.token_vocab)

        self.vocab_mapper.pad_idx = self.vocab_mapper.token_vocab.get(self.vocab_mapper.pad_token, 0)
        self.vocab_mapper.unk_idx = self.vocab_mapper.token_vocab[self.vocab_mapper.unk_token]

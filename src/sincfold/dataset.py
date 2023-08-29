import pandas as pd
from torch.utils.data import Dataset
import torch as tr
import os
import json
import pickle
from sincfold.embeddings import OneHotEmbedding
from sincfold.utils import valid_mask, prob_mat, bp2matrix, dot2bp


class SeqDataset(Dataset):
    def __init__(
        self, dataset_path, min_len=0, max_len=512, verbose=False, cache_path=None, for_prediction=False, 
        use_restrictions=False, **kargs):
        self.max_len = max_len
        self.verbose = verbose
        if cache_path is not None and not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache = cache_path

        # Loading dataset
        data = pd.read_csv(dataset_path)
        
        if for_prediction:
            assert (
                "sequence" in data.columns
                and "id" in data.columns
            ), "Dataset should contain 'id' and 'sequence' columns"

        else:
            assert (
                ("base_pairs" in data.columns or "dotbracket" in data.columns)
                and "sequence" in data.columns
                and "id" in data.columns
            ), "Dataset should contain 'id', 'sequence' and 'base_pairs' or 'dotbracket' columns"

            if "base_pairs" not in data.columns and "dotbracket" in data.columns:
                data["base_pairs"] = data.dotbracket.apply(lambda x: str(dot2bp(x)))      

        data["len"] = data.sequence.str.len()

        if max_len is None:
            max_len = max(data.len)
        self.max_len = max_len

        datalen = len(data)

        data = data[(data.len >= min_len) & (data.len <= max_len)]

        if len(data) < datalen:
            print(
                f"From {datalen} sequences, filtering {min_len} < len < {max_len} we have {len(data)} sequences"
            )

        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()

        self.embedding = OneHotEmbedding()
        self.embedding_size = self.embedding.emb_size
        self.use_restrictions = use_restrictions

        self.base_pairs = None
        if "base_pairs" in data.columns:
            self.base_pairs = [
                json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
            ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqid = self.ids[idx]
        cache = f"{self.cache}/{seqid}.pk"
        if (self.cache is not None) and os.path.isfile(cache):
            item = pickle.load(open(cache, "rb"))
        else:
            sequence = self.sequences[idx]
            L = len(sequence)
            Mc = None
            if self.base_pairs is not None:
                Mc = bp2matrix(L, self.base_pairs[idx])

            seq_emb = self.embedding.seq2emb(sequence)

            mask = valid_mask(sequence)
            if self.use_restrictions:
                prob_mask = prob_mat(sequence)
                item = [seq_emb, Mc, L, mask, seqid, sequence, prob_mask]
            else:
                item = [seq_emb, Mc, L, mask, seqid, sequence]
                

            if self.cache is not None:
                pickle.dump(item, open(cache, "wb"))
                
        return item


def pad_batch(batch):
    """batch is a list of (seq_emb, Mc, L, mask, prob_mask, seqid)"""
    if len(batch[0]) == 7:
        seq_emb, Mc, L, mask, seqid, sequence, prob_mask = zip(*batch)
    else:
        prob_mask = None
        seq_emb, Mc, L, mask, seqid, sequence = zip(*batch)

    seq_emb_pad = tr.zeros((len(batch), seq_emb[0].shape[0], max(L)))
    if Mc[0] is None:
        Mc_pad = None
    else:
        Mc_pad = -tr.ones((len(batch), max(L), max(L)), dtype=tr.long)
    mask_pad = tr.zeros((len(batch), max(L), max(L)))
    if prob_mask is not None:
        prob_mask_pad = tr.zeros((len(batch), max(L), max(L)))

    for k in range(len(batch)):
        seq_emb_pad[k, :, : L[k]] = seq_emb[k]
        if Mc_pad is not None:
            Mc_pad[k, : L[k], : L[k]] = Mc[k]
        mask_pad[k, : L[k], : L[k]] = mask[k]
        if prob_mask is not None:
            prob_mask_pad[k, : L[k], : L[k]] = prob_mask[k]

    if prob_mask is not None:
        return seq_emb_pad, Mc_pad, L, mask_pad, seqid, sequence, prob_mask_pad
    else:
        return seq_emb_pad, Mc_pad, L, mask_pad, seqid, sequence

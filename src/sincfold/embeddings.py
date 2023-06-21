import torch as tr

# Mapping of nucleotide symbols
# R	Guanine / Adenine (purine)
# Y	Cytosine / Uracil (pyrimidine)
# K	Guanine / Uracil
# M	Adenine / Cytosine
# S	Guanine / Cytosine
# W	Adenine / Uracil
# B	Guanine / Uracil / Cytosine
# D	Guanine / Adenine / Uracil
# H	Adenine / Cytosine / Uracil
# V	Guanine / Cytosine / Adenine
# N	Adenine / Guanine / Cytosine / Uracil
NT_DICT = {
    "R": ["G", "A"],
    "Y": ["C", "U"],
    "K": ["G", "U"],
    "M": ["A", "C"],
    "S": ["G", "C"],
    "W": ["A", "U"],
    "B": ["G", "U", "C"],
    "D": ["G", "A", "U"],
    "H": ["A", "C", "U"],
    "V": ["G", "C", "A"],
    "N": ["G", "A", "C", "U"],
}

VOCABULARY = ["A", "C", "G", "U"]


class OneHotEmbedding:
    def __init__(self):
        self.pad_token = "-"
        self.vocabulary = VOCABULARY
        self.emb_size = len(self.vocabulary)

    def seq2emb(self, seq, pad_token="-"):
        """One-hot representation of seq nt in vocabulary.  Emb is CxL
        Other nt are mapped as shared activations.
        """
        seq = seq.upper().replace("T", "U")  # convert to RNA
        emb_size = len(VOCABULARY)
        emb = tr.zeros((emb_size, len(seq)), dtype=tr.float)

        for k, nt in enumerate(seq):
            if nt == pad_token:
                continue
            if nt in VOCABULARY:
                emb[VOCABULARY.index(nt), k] = 1
            elif nt in NT_DICT:
                v = 1 / len(NT_DICT[nt])
                ind = [VOCABULARY.index(n) for n in NT_DICT[nt]]
                emb[ind, k] = v
            else:
                raise ValueError(f"Unrecognized nucleotide {nt}")

        return emb

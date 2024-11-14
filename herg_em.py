from Bio import SeqIO
from Bio import ExPASy

def fetch_sequence(uniprot_id):
    try:
        handle = ExPASy.get_sprot_raw(uniprot_id)
        record = SeqIO.read(handle, "swiss")
        sequence = str(record.seq)
        return sequence
    except Exception as e:
        print("Error fetching sequence:", e)
        return None

import torch
import esm

def cal_em(sequence):
# Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("hERG", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    return sequence_representations

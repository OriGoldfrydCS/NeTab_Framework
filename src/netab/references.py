"""Independent position donors, shared by class, seed and target length."""

import numpy as np


def reference_bank(proposal, label, lengths, seed, m, dataset_id=1, class_index=0):
    p = proposal.schema.positions
    bank = np.full((p + 1, m, p), -1, np.int32)
    for length in np.unique(lengths):
        for ref in range(m):
            for j in range(int(length)):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [8301, class_index, seed, int(length), j + ref * 100000, dataset_id]
                    )
                )
                bank[length, ref, j] = rng.choice(proposal.donors(label, j))
    return bank

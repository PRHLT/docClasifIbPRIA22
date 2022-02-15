from __future__ import print_function
from __future__ import division
from builtins import range
import numpy as np

# --- Lavenshtein edit distance
def levenshtein(hyp, target):
    """
    levenshtein edit distance using
    addcost=delcost=subcost=1
    Borrowed form: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(hyp) < len(target):
        return levenshtein(target, hyp)

    # So now we have len(hyp) >= len(target).
    if len(target) == 0:
        return len(hyp)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    hyp = np.array(tuple(hyp))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in hyp:
        # Insertion (target grows longer than hyp):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and hyp items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:], np.add(previous_row[:-1], target != s)
        )

        # Deletion (target grows shorter than hyp):
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

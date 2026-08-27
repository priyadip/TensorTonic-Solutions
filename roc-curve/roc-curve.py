import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by descending score.
    order = np.argsort(-y_score)
    scores = y_score[order]
    labels = y_true[order]

    P = np.sum(labels == 1)
    N = np.sum(labels == 0)

    tp = 0
    fp = 0

    fpr = [0.0]
    tpr = [0.0]
    thresholds = [np.inf]

    i = 0
    n = len(scores)

    while i < n:
        # Group all samples having the same score.
        score = scores[i]
        j = i

        while j < n and scores[j] == score:
            if labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        fpr.append(float(fp / N))
        tpr.append(float(tp / P))
        thresholds.append(float(score))

        i = j

    return {
        "fpr": np.asarray(fpr),
        "tpr": np.asarray(tpr),
        "thresholds": np.asarray(thresholds)
    }
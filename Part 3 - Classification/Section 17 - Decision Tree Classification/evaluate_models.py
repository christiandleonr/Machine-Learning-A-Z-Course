def evaluate(tp, tn, fp, fn):
    if not isinstance(tp, int) or not isinstance(tn, int) or not isinstance(fp, int) or not isinstance(fn, int):
        raise ValueError

    accuracy = (tp+tn)/(tp+tn+fp+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return accuracy, precision, recall
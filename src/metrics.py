from sklearn.metrics import f1_score, precision_score, recall_score

def get_metric(metric_name, labels, predictions, threshold=0.5, average='macro'):
    binary_predictions = (predictions > threshold).astype(float)
    if metric_name == 'f1_score':
        return f1_score(labels, binary_predictions, average=average, zero_division=0)
    elif metric_name == 'recall':
        return recall_score(labels, binary_predictions, average=average, zero_division=0)
    elif metric_name == 'precision':
        return precision_score(labels, binary_predictions, average=average, zero_division=0)
    else:
        raise 'Please, specify a valid metric. One of f1_score, recall, precision'
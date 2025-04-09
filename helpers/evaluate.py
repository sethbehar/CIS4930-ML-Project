from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_true, pred):
    metrics = []
    
    metrics.append(accuracy_score(y_true, pred))
    metrics.append(precision_score(y_true, pred))
    metrics.append(recall_score(y_true, pred))
    metrics.append(f1_score(y_true, pred))
        
    return metrics


def print_stats(metrics, name):
    print(name, " Accuracy Score: ", metrics[0])
    print(name, " Precision Score: ", metrics[1])
    print(name, " Recalle Score: ", metrics[2])
    print(name, " F1 Score: ", metrics[3])

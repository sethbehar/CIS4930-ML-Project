import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def evaluate_model(y_true, y_pred, model_name="Model"):
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print(f"{model_name} Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Return metrics as a dictionary
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
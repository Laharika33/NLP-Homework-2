import numpy as np

# -----------------------------------------------------------
# Confusion matrix setup
# Rows = system predictions, Columns = actual gold labels
# Example: matrix[0,1] = number of items predicted as "Cat" 
# but actually "Dog"
# -----------------------------------------------------------
conf_matrix = np.array([
    [5, 10, 5],    # Predicted as Cat
    [15, 20, 10],  # Predicted as Dog
    [0, 15, 10]    # Predicted as Rabbit
])

# Number of classes and class labels
num_classes = conf_matrix.shape[0]
classes = ['Cat', 'Dog', 'Rabbit']

# -----------------------------------------------------------
# Per-class precision and recall
# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)
# -----------------------------------------------------------
precision = []
recall = []

for i in range(num_classes):
    # True Positives = correct predictions for this class
    TP = conf_matrix[i, i]
    
    # False Positives = predicted as this class but actually other classes
    FP = conf_matrix[i, :].sum() - TP
    
    # False Negatives = actually this class but predicted as others
    FN = conf_matrix[:, i].sum() - TP
    
    # Compute precision and recall safely (avoid division by zero)
    p = TP / (TP + FP) if (TP + FP) != 0 else 0
    r = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    precision.append(p)
    recall.append(r)
    
    print(f"{classes[i]} -> Precision: {p:.3f}, Recall: {r:.3f}")

# -----------------------------------------------------------
# Macro-averaged metrics
# (Treat each class equally by averaging their metrics)
# -----------------------------------------------------------
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)

print(f"\nMacro-averaged Precision: {macro_precision:.3f}")
print(f"Macro-averaged Recall: {macro_recall:.3f}")

# -----------------------------------------------------------
# Micro-averaged metrics
# (Aggregate counts across all classes first, then compute metrics)
# -----------------------------------------------------------

# Total True Positives = sum of diagonal elements
TP_total = np.trace(conf_matrix)

# Total False Positives = all predictions - correct ones
FP_total = conf_matrix.sum(axis=1).sum() - TP_total

# Total False Negatives = all actual labels - correct ones
FN_total = conf_matrix.sum(axis=0).sum() - TP_total

# Compute micro precision and recall
micro_precision = TP_total / (TP_total + FP_total)
micro_recall = TP_total / (TP_total + FN_total)

print(f"\nMicro-averaged Precision: {micro_precision:.3f}")
print(f"Micro-averaged Recall: {micro_recall:.3f}")

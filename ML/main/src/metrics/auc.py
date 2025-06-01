# From https://claude.ai/chat/45f57179-d650-49cc-a4da-b6c905263b34

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def calculate_auc(y_true, y_pred_proba):
    """
    Calculate the Area Under the ROC Curve (AUC) score.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
        
    Returns:
    --------
    float
        AUC score.
    """
    # Input validation
    if len(y_true) != len(y_pred_proba):
        raise ValueError("Length of y_true and y_pred_proba must be the same")
    
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Validate binary classification
    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2:
        raise ValueError(f"Expected binary classification with 2 classes, got {len(unique_classes)} classes")
    
    # Calculate AUC score
    auc_score = roc_auc_score(y_true, y_pred_proba)
    return auc_score

def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 6)):
    """
    Plot the ROC curve and calculate AUC.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
    figsize : tuple, optional
        Figure size (width, height) in inches.
        
    Returns:
    --------
    tuple
        Figure and AUC score.
    """
    # Calculate AUC
    auc_score = calculate_auc(y_true, y_pred_proba)
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    return plt.gcf(), auc_score

# Custom implementation of AUC without using sklearn
def calculate_auc_manual(y_true, y_pred_proba):
    """
    Manual implementation of AUC calculation without using sklearn.
    Uses the Mann-Whitney U statistic interpretation of AUC.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
        
    Returns:
    --------
    float
        AUC score.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Get indices of positive and negative examples
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    # If there are no positive or negative examples, AUC is undefined
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        raise ValueError("AUC is undefined when there is only one class present")
    
    # Count the number of pairs where positive examples have higher scores than negative examples
    n_pairs = 0
    n_correct_pairs = 0
    
    # For each positive example
    for pos_idx in pos_indices:
        pos_score = y_pred_proba[pos_idx]
        
        # Compare with each negative example
        for neg_idx in neg_indices:
            neg_score = y_pred_proba[neg_idx]
            n_pairs += 1
            
            if pos_score > neg_score:
                n_correct_pairs += 1
            elif pos_score == neg_score:
                # In case of ties, count as 0.5 correct
                n_correct_pairs += 0.5
    
    # AUC is the fraction of pairs that are correctly ordered
    auc_score = n_correct_pairs / n_pairs
    return auc_score

# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    n_samples = 100
    
    # True labels (0 or 1)
    y_true = np.random.randint(0, 2, n_samples)
    
    # Predicted probabilities (biased towards correct classification)
    y_pred_good = np.random.beta(8, 2, n_samples) * y_true + np.random.beta(2, 8, n_samples) * (1 - y_true)
    y_pred_random = np.random.random(n_samples)  # Random predictions
    
    # Calculate AUC using sklearn
    auc_good = calculate_auc(y_true, y_pred_good)
    auc_random = calculate_auc(y_true, y_pred_random)
    
    # Calculate AUC using manual implementation
    auc_manual_good = calculate_auc_manual(y_true, y_pred_good)
    auc_manual_random = calculate_auc_manual(y_true, y_pred_random)
    
    print(f"Good model - AUC (sklearn): {auc_good:.3f}")
    print(f"Good model - AUC (manual): {auc_manual_good:.3f}")
    print(f"Random model - AUC (sklearn): {auc_random:.3f}")
    print(f"Random model - AUC (manual): {auc_manual_random:.3f}")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_good)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_good:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Good Model')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_pred_random)
    plt.plot(fpr, tpr, color='red', lw=2, label=f'AUC = {auc_random:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Model')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
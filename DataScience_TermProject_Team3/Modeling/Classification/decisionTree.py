import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def run_decisionTree(csv_path: str, KFold=10, depth = 3):
    #Load dataset
    df = pd.read_csv(csv_path)

    #Selection Features (based on Feature selection) and target
    features = [
        'word_count', 'char_length', 'digit_count', 'uppercase_count',
        'special_char_count', 'has_url', 'has_keywords',
        'digit_ratio', 'uppercase_ratio', 'spam_keyword_count'
    ]
    X = df[features]
    y = df['label']

    #Split train/test set (Stratify)
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y, test_size = 0.2, stratify=y, random_state=42
    )

    #Stratified KFold CV (Imbalanced labed)
    skf = StratifiedKFold(KFold, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []

    #(KFoldEvaluation) Training Model with DecisionTreeClassifier (depth = 3 (Prevent Overfitting))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = DecisionTreeClassifier(max_depth = depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        #Store results
        acc_list.append(acc)
        f1_list.append(f1)

        print(f"\n Fold {fold+1}'s Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    #Printing Average Accuracy and F1 Score
    print("\n K-Fold Cross Validation Results (K=10):")
    print(f"Average Accuracy: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Average F1 Score: {sum(f1_list)/len(f1_list):.4f}")

    #Final model training and Predict with final test set
    final_model = DecisionTreeClassifier(max_depth = depth, random_state=42)
    final_model.fit(X_train_full, y_train_full)
    y_pred_final = final_model.predict(X_test_final)

    print("\n Final Test Set Evaluation:")
    print(classification_report(y_test_final, y_pred_final, target_names=["Ham", "Spam"]))

    #Confusion Matrix
    cm = confusion_matrix(y_test_final, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title("Confusion Matrix - Decision Tree (Final Test)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    #Tree visualization
    plt.figure(figsize=(16, 10))
    plot_tree(final_model, feature_names=X.columns, class_names=["Ham", "Spam"], filled=True, max_depth=3)
    plt.title(f"Decision Tree Visualization (max_depth={depth})")
    plt.tight_layout()
    plt.show()
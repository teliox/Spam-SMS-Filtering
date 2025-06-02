import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def run_NaiveBayes_TF_IDF(csv_path: str, KFold = 5):
    #Load dataset
    df = pd.read_csv(csv_path)

    #Extracting TF-IDF features (strat with tfidf_)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    X = df[tfidf_cols]
    y = df["label"]

    #Spliting Train/Final test dataset (stratify)
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y, test_size = 0.2 , stratify=y, random_state=42
    )

    #StratifiedKFold Cross validation
    skf = StratifiedKFold(KFold, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        acc_list.append(acc)
        f1_list.append(f1)

        print(f"\n Fold {fold+1} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    #Printing Average scores
    print("\n K-Fold Cross Validation Results:")
    print(f"Average Accuracy: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Average F1 Score: {sum(f1_list)/len(f1_list):.4f}")

    #Final model training and test with final test dataset
    final_model = MultinomialNB()
    final_model.fit(X_train_full, y_train_full)
    y_pred_final = final_model.predict(X_test_final)

    print("\n Final Test Set Evaluation:")
    print(classification_report(y_test_final, y_pred_final))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_final, y_pred_final)
    print(cm)

    #Visualization Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (TF-IDF + NaiveBayes Final Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    #https://github.com/arsenyturin/nlp_multinomial_naive_bayes
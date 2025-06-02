import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix,
                             roc_curve, auc, RocCurveDisplay, make_scorer)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

def run_RandomForest_Eval(csv_path: str, KFold = 5):
    #Load CSV file
    df = pd.read_csv(csv_path)

    #Define features and targets
    feature_cols = [
        'word_count', 'char_length', 'digit_count', 'uppercase_count',
        'special_char_count', 'digit_ratio', 'uppercase_ratio',
        'has_url', 'has_keywords', 'spam_keyword_count'
    ]
    X = df[feature_cols]
    y = df['label']

    #Split train/test set (Stratify)
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    #Stratified KFold CV (Imbalanced labed)
    cv_strategy = StratifiedKFold(KFold, shuffle=True, random_state=42)

    #Define F1 scorer about Spam(minority Class)
    f1_spam_scorer = make_scorer(f1_score, pos_label=1, average='binary') 


    #Find optimized Hyperparameter with using RandomizedSearchCV
    print("\n--- RandomizedSearchCV for RandomForest ---")

    #RandomizedSearchCV
    param_dist_rf = {
        'n_estimators': randint(50, 201),
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': randint(2, 21),
        'min_samples_leaf': randint(1, 11),
        'max_features': ['sqrt', 'log2', 0.5, 0.7, None],
        'criterion': ['gini', 'entropy'],
    }

    rf_model_for_tuning = RandomForestClassifier(random_state=42)

    #n_iterations : number of parameter combinations to try
    n_iterations = 30 
    random_search_rf = RandomizedSearchCV(estimator=rf_model_for_tuning,
                                        param_distributions=param_dist_rf,
                                        n_iter=n_iterations,
                                        cv=cv_strategy,
                                        scoring=f1_spam_scorer, 
                                        verbose=1,
                                        n_jobs=-1,
                                        random_state=42)

    random_search_rf.fit(X_train_full, y_train_full)
    
    top_n = 5
    results_df = pd.DataFrame(random_search_rf.cv_results_)

    #Sorting Top 5 results order of Score
    top_results = results_df.sort_values(by='mean_test_score', ascending=False).head(top_n)

    #Printing result
    print(f"\nTop {top_n} Hyperparameter Combinations (sorted by F1 Score on Spam):")
    for i, row in top_results.iterrows():
        print(f"\nRank {i+1}")
        print(f"F1 Score: {row['mean_test_score']:.4f}")
        print(f"Params: {row['params']}")

    print(f"\n Best Hyperparameters with RandomizedSearchCV :")
    print(random_search_rf.best_params_)
    print(f"Best F1-score (Spam) from RandomizedSearchCV : {random_search_rf.best_score_:.4f}") 

    #Decide Final model with optimized hyperparameter
    final_model_rf = random_search_rf.best_estimator_
    y_pred_final_rf = final_model_rf.predict(X_test_final)


    #Final evaluation of optimized model
    print("\n Final Test Set Evaluation :")
    print(classification_report(y_test_final, y_pred_final_rf, target_names=["Ham (0)", "Spam (1)"]))

    #Confusion Matrix
    cm_rf = confusion_matrix(y_test_final, y_pred_final_rf)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title("Confusion Matrix - RandomForest (Final Test, Best Params)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    #ROC Curve and AUC
    metrics_rf = {}
    report_rf_dict = classification_report(y_test_final, y_pred_final_rf, output_dict=True, target_names=["Ham", "Spam"]) 
    metrics_rf["Accuracy"] = accuracy_score(y_test_final, y_pred_final_rf)
    metrics_rf["F1 (Spam)"] = report_rf_dict["Spam"]["f1-score"] 
    metrics_rf["Recall (Spam)"] = report_rf_dict["Spam"]["recall"] 
    metrics_rf["Precision (Spam)"] = report_rf_dict["Spam"]["precision"] 
    metrics_rf["F1 (Ham)"] = report_rf_dict["Ham"]["f1-score"] 
    metrics_rf["Macro Avg F1"] = report_rf_dict["macro avg"]["f1-score"]

    #Prediction of spam (class 1) probabilities for test sets
    y_pred_proba_final_rf = final_model_rf.predict_proba(X_test_final)[:, 1] 
    #FPR, TPR calculation for ROC curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test_final, y_pred_proba_final_rf, pos_label=1) 
    
    #AUC (Area Under the Curve) Calculation
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    metrics_rf["AUC (Spam)"] = roc_auc_rf 
    print(f"\nAUC (Spam): {roc_auc_rf:.4f}") 

    #ROC Curve Visualization
    plt.figure(figsize=(8, 6))
    disp_rf = RocCurveDisplay(fpr=fpr_rf, tpr=tpr_rf, roc_auc=roc_auc_rf, estimator_name='Best RandomForest')
    disp_rf.plot(ax=plt.gca())
    #Random guess baseline (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5)')
    plt.title('ROC Curve - RandomForest (Final Test, Best Params, Target: Spam)') 
    plt.legend()
    plt.show()


    #Print Score metric
    print("\n--- Calculated Metrics ---")
    for metric_name, score in metrics_rf.items():
        print(f"{metric_name}: {score:.4f}")

    #https://zephyrus1111.tistory.com/425


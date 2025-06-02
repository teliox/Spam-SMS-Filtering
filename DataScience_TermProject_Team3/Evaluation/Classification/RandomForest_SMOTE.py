import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix,
                             roc_curve, auc, RocCurveDisplay, make_scorer)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def run_RandomForest_SMOTE(csv_path: str, KFold = 5):
    #Load dataset
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


    # --- Case 1: RandomizedSearchCV without SMOTE ---
    print("\n--- RandomizedSearchCV for RandomForest (without SMOTE) ---")

    param_dist_rf_no_smote = {
        'n_estimators': randint(50, 201),
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': randint(2, 21),
        'min_samples_leaf': randint(1, 11),
        'max_features': ['sqrt', 'log2', 0.6, 0.8],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', 'balanced_subsample', {0:1, 1:5}, {0:1, 1:10}]
    }

    #n_iterations : number of parameter combinations to try
    rf_model_no_smote_tuning = RandomForestClassifier(random_state=42)
    n_iterations_no_smote = 30

    random_search_rf_no_smote = RandomizedSearchCV(estimator=rf_model_no_smote_tuning,
                                                param_distributions=param_dist_rf_no_smote,
                                                n_iter=n_iterations_no_smote,
                                                cv=cv_strategy,
                                                scoring=f1_spam_scorer,
                                                verbose=1,
                                                n_jobs=-1,
                                                random_state=42)
    random_search_rf_no_smote.fit(X_train_full, y_train_full)
    final_model_rf_no_smote = random_search_rf_no_smote.best_estimator_
    y_pred_final_rf_no_smote = final_model_rf_no_smote.predict(X_test_final)
    
    top_n = 5
    results_df = pd.DataFrame(random_search_rf_no_smote.cv_results_)

    #Sorting Top 5 results order of Score
    top_results = results_df.sort_values(by='mean_test_score', ascending=False).head(top_n)

    #Printing result
    print(f"\nTop {top_n} Hyperparameter Combinations (sorted by F1 Score on Spam):")
    for i, row in top_results.iterrows():
        print(f"\nRank {i+1}")
        print(f"F1 Score: {row['mean_test_score']:.4f}")
        print(f"Params: {row['params']}")

    print(f"\n Best Hyperparameters (RandomForest without SMOTE) :")
    print(random_search_rf_no_smote.best_params_)
    print(f"Best F1 (Spam) from CV (RandomForest without SMOTE): {random_search_rf_no_smote.best_score_:.4f}")

    #To Store scores without SMOTE
    report_rf_no_smote_dict = classification_report(y_test_final, y_pred_final_rf_no_smote, output_dict=True, target_names=["Ham", "Spam"])
    metrics_rf_no_smote = {
        "Accuracy": accuracy_score(y_test_final, y_pred_final_rf_no_smote),
        "F1 (Spam)": report_rf_no_smote_dict["Spam"]["f1-score"],
        "Recall (Spam)": report_rf_no_smote_dict["Spam"]["recall"],
        "Precision (Spam)": report_rf_no_smote_dict["Spam"]["precision"],
        "F1 (Ham)": report_rf_no_smote_dict["Ham"]["f1-score"],
        "Macro Avg F1": report_rf_no_smote_dict["macro avg"]["f1-score"]
    }

    #ROC and AUC (Spam is positive)
    y_pred_proba_rf_no_smote = final_model_rf_no_smote.predict_proba(X_test_final)[:, 1]
    fpr_rf_ns, tpr_rf_ns, _ = roc_curve(y_test_final, y_pred_proba_rf_no_smote, pos_label=1)
    metrics_rf_no_smote["AUC (Spam)"] = auc(fpr_rf_ns, tpr_rf_ns)


    #Print result of Final Test set without SMOTE
    print("\n Final Test Set Evaluation (RandomForest without SMOTE):")
    print(classification_report(y_test_final, y_pred_final_rf_no_smote, target_names=["Ham (0)", "Spam (1)"]))
    if not pd.isna(metrics_rf_no_smote["AUC (Spam)"]):
        print(f"AUC (Spam): {metrics_rf_no_smote['AUC (Spam)']:.4f}")


    # --- Case 2 : RandomizedSearchCV with using SMOTE---
    print("\n--- RandomizedSearchCV for RandomForest (with SMOTE) ---")

    #Pipelining SMOTE and RandomForestClassifier
    #Exploring smote_k_neighbors and classifier parameters together
    #*Pipeline is a tool that sequentially groups multiple machine learning processing steps.
    pipeline_smote_rf = ImbPipeline([
        ('smote', SMOTE(random_state=42)), #SMOTE oversampling for data imbalance
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    #Dictionary for parameter for pipeline
    param_dist_smote_rf = {
        'classifier__n_estimators': randint(50, 201),
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__min_samples_split': randint(2, 21),
        'classifier__min_samples_leaf': randint(1, 11),
        'classifier__max_features': ['sqrt', 'log2', 0.6, 0.8],
        'classifier__criterion': ['gini', 'entropy']
    }
    n_iterations_smote = 30

    #RandomizedSearchCV
    #f1_spam_scorer: a custom f1-score scorer focused on the Spam(=1) class.
    random_search_smote_rf = RandomizedSearchCV(estimator=pipeline_smote_rf,
                                                param_distributions=param_dist_smote_rf,
                                                n_iter=n_iterations_smote,
                                                cv=cv_strategy,
                                                scoring=f1_spam_scorer,
                                                verbose=1,
                                                n_jobs=-1,
                                                random_state=42)

    #Result
    random_search_smote_rf.fit(X_train_full, y_train_full)

    # --- Visualize the distribution of training data labels after applying SMOTE ---
    #fit_resample(): actually oversampling data.
    best_pipeline_smote_rf = random_search_smote_rf.best_estimator_
    smote_step = best_pipeline_smote_rf.named_steps['smote']
    X_train_smoted, y_train_smoted = smote_step.fit_resample(X_train_full, y_train_full)

    #Visualization Label Distribution : Before SMOTE vs After SMOTE-------------------------
    print("\n--- Label Distribution Before and After SMOTE (on Training Data) ---")
    print("Original training data distribution:")
    original_counts = pd.Series(y_train_full).value_counts().sort_index()
    print(original_counts)

    print("\nTraining data distribution after SMOTE:")
    smoted_counts = pd.Series(y_train_smoted).value_counts().sort_index()
    print(smoted_counts)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x=original_counts.index, y=original_counts.values, palette=['#4CAF50', '#F44336'])
    plt.title('Label Distribution in Original Training Data')
    plt.xlabel('Label (0: Ham, 1: Spam)')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Ham (0)', 'Spam (1)'])
    for i, count in enumerate(original_counts):
        plt.text(i, count + 5, str(count), ha='center', va='bottom')


    plt.subplot(1, 2, 2)
    sns.barplot(x=smoted_counts.index, y=smoted_counts.values, palette=['#4CAF50', '#F44336'])
    plt.title('Label Distribution in Training Data After SMOTE')
    plt.xlabel('Label (0: Ham, 1: Spam)')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Ham (0)', 'Spam (1)'])
    for i, count in enumerate(smoted_counts):
        plt.text(i, count + 5, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
    #-----------------------------------------------------------------------------------

    #Performs predictions on the test set
    final_model_smote_rf = best_pipeline_smote_rf
    y_pred_final_smote_rf = final_model_smote_rf.predict(X_test_final)
    
    top_n = 5
    results_df = pd.DataFrame(random_search_smote_rf.cv_results_)

    #Sorting Top 5 results order of Score
    top_results = results_df.sort_values(by='mean_test_score', ascending=False).head(top_n)

    #Printing result
    print(f"\nTop {top_n} Hyperparameter Combinations (sorted by F1 Score on Spam):")
    for i, row in top_results.iterrows():
        print(f"\nRank {i+1}")
        print(f"F1 Score: {row['mean_test_score']:.4f}")
        print(f"Params: {row['params']}")


    print(f"\n Best Hyperparameters (RandomForest with SMOTE) :")
    print(random_search_smote_rf.best_params_)
    print(f"Best F1 (Spam) from CV (RandomForest with SMOTE): {random_search_smote_rf.best_score_:.4f}")

    #Store Scores with using SMOTE
    #Precision, Recall, F1, Accuracy, Macro F1 calculation for each class
    report_smote_rf_dict = classification_report(y_test_final, y_pred_final_smote_rf, output_dict=True, target_names=["Ham", "Spam"])
    metrics_rf_smote = {
        "Accuracy": accuracy_score(y_test_final, y_pred_final_smote_rf),
        "F1 (Spam)": report_smote_rf_dict["Spam"]["f1-score"],
        "Recall (Spam)": report_smote_rf_dict["Spam"]["recall"],
        "Precision (Spam)": report_smote_rf_dict["Spam"]["precision"],
        "F1 (Ham)": report_smote_rf_dict["Ham"]["f1-score"],
        "Macro Avg F1": report_smote_rf_dict["macro avg"]["f1-score"]
    }

    #ROC and AUC
    #AUC is an indicator that comprehensively measures discrimination ability in class classification

    y_pred_proba_smote_rf = final_model_smote_rf.predict_proba(X_test_final)[:, 1]
    fpr_s_rf, tpr_s_rf, _ = roc_curve(y_test_final, y_pred_proba_smote_rf, pos_label=1)
    metrics_rf_smote["AUC (Spam)"] = auc(fpr_s_rf, tpr_s_rf)


    #Final result with using SMOTE
    print("\n Final Test Set Evaluation (RandomForest with SMOTE):")
    print(classification_report(y_test_final, y_pred_final_smote_rf, target_names=["Ham (0)", "Spam (1)"]))
    if not pd.isna(metrics_rf_smote["AUC (Spam)"]):
        print(f"AUC (Spam, with SMOTE): {metrics_rf_smote['AUC (Spam)']:.4f}")

    # --- ROC Curve Visualization for SMOTE-based RandomForest ---
    plt.figure(figsize=(8, 6))
    disp_smote_rf = RocCurveDisplay(
        fpr=fpr_s_rf,
        tpr=tpr_s_rf,
        roc_auc=metrics_rf_smote["AUC (Spam)"],
        estimator_name='RandomForest + SMOTE'
    )
    disp_smote_rf.plot(ax=plt.gca())
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5)')
    plt.title('ROC Curve - RandomForest with SMOTE (Target: Spam)', fontsize=14)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    # --- Visualization difference Before SMOTE vs After SMOTE ---
    print("\nDifference Before and After SMOTE")
    metric_names_rf = ["Accuracy", "F1 (Spam)", "Recall (Spam)", "Precision (Spam)", "Macro Avg F1", "AUC (Spam)", "F1 (Ham)"]
    no_smote_values_rf = [metrics_rf_no_smote.get(metric, float('nan')) for metric in metric_names_rf]
    smote_values_rf = [metrics_rf_smote.get(metric, float('nan')) for metric in metric_names_rf]

    #Compare performance metrics in bar form
    comparison_df_rf = pd.DataFrame({
        'Metric': metric_names_rf,
        'Without SMOTE': no_smote_values_rf,
        'With SMOTE': smote_values_rf
    }).fillna(0)

    comparison_df_rf_melted = comparison_df_rf.melt(id_vars='Metric', var_name='Model Type', value_name='Score')

    #Visualization Comparison
    plt.figure(figsize=(15, 9))
    sns.barplot(x='Metric', y='Score', hue='Model Type', data=comparison_df_rf_melted, palette={'Without SMOTE': '#2196F3', 'With SMOTE': '#FF9800'})
    plt.title('Performance Comparison: RandomForest Before and After SMOTE (Target: Spam)', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title='Model Type', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    for p in plt.gca().patches:  #.patches : each bar in a bar graph
        if p.get_height() > 0: 
            plt.gca().annotate(f"{p.get_height():.3f}",  #plt.gca() returns the currently used subplot(axes)
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 8),
                            textcoords='offset points')
    plt.show()

    #https://yumdata.tistory.com/383
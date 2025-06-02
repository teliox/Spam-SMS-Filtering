import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_FeatureSelection(csv_path: str):
    #Load dataset
    df = pd.read_csv(csv_path)
    y = df['label']

    #Numerical features
    numeric_features = [
        'word_count', 'char_length', 'digit_count', 'uppercase_count',
        'special_char_count', 'digit_ratio', 'uppercase_ratio', 'spam_keyword_count'
    ]

    #Binary features
    binary_features = ['has_url', 'has_keywords']

    #Encoded_catrgorical features
    categorical_features = ['starting_word_encoded', 'length_bin_encoded', 'digit_bin_encoded']

    #All Feature sets
    all_features = numeric_features + binary_features + categorical_features
    X = df[all_features]


    #SelectKBest (Univariate Selection)
    selector = SelectKBest(score_func=f_classif, k=8)
    selector.fit(X, y)
    selected_kbest = X.columns[selector.get_support()].tolist()

    #Visualization
    kbest_scores = selector.scores_[selector.get_support()]
    kbest_features = X.columns[selector.get_support()]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=kbest_scores, y=kbest_features, palette='Blues_d')
    plt.title("Top 8 Features by SelectKBest")
    plt.xlabel("Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    #RandomForest Feature Importance
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X, y)
    importances = forest.feature_importances_
    forest_ranking = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    selected_rf = forest_ranking.head(8).index.tolist()

    #Visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(x=forest_ranking.head(8).values, y=forest_ranking.head(8).index, palette='Greens_d')
    plt.title("Top 8 Features by RandomForest Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    #Correlation Heatmap
    corr_matrix = df[all_features + ['label']].corr()
    corr_with_label = corr_matrix['label'].drop('label').abs().sort_values(ascending=False)
    selected_corr = corr_with_label.head(8).index.tolist()

    #Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.loc[selected_corr + ['label'], selected_corr + ['label']], annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap (Top 8 with label)")
    plt.tight_layout()
    plt.show()

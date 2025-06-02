import os
from DataExploration.DataExploration import run_exploration
from DataPreprocessing.Preprocessing import run_Preprocessing
from DataExploration.DataExploration_AfterPreprocessing import run_DataExploration_AfterPreprocessing
from DataPreprocessing.FeatureSelection import run_FeatureSelection
from Modeling.Classification.decisionTree import run_decisionTree
from Modeling.Classification.NaiveBayes_TF_IDF import run_NaiveBayes_TF_IDF
from Modeling.Classification.RandomForest import run_RandomForest
from Modeling.kmeansClustering.SpamCluster import run_SpamCluster
from Evaluation.Classification.RandomForest_SMOTE import run_RandomForest_SMOTE
from Evaluation.Classification.RandomForest import run_RandomForest_Eval
from Evaluation.kmeansClustering.KMeans_Eval import run_KMeans_Eval


def main():
    #Dataset paths
    csv_path = "spam.csv"
    label_messsage_path = "df_label_message_only.csv"

    #Data Exploration
    run_exploration(csv_path)

    #Data Preprocessing
    #Scaler = 'Standard' or 'Robust', Encoding = True or False, max_features : Feature dimension of TF-IDF
    Scaler : str
    Scaler = input("Select Scaling method (Standard / Robust) : ").strip().lower()

    if Scaler not in ['standard', 'robust']:
        print("Invalid Value entered")
        exit(1)
    
    max_features = int(input("Select TF-IDF Feature Dimension (150 / 300 / 450 / 600) : "))
    
    # Encoding = int(input("Select Encoding option (On : 1 / Off : 0) : "))
    # if(Encoding != 1 or Encoding != 0) :
    #     print("Invalid Value entered")

    run_Preprocessing(csv_path, Scaler, max_features)

    #Path of preprocessed data file
    preprocessed_path_tfidf = os.path.join(os.path.dirname(csv_path), "spam_with_tfidf.csv")
    preprocessed_path_no_tfidf = os.path.join(os.path.dirname(csv_path), "spam_no_tfidf.csv")

    #Data Exploration after preprocessing
    run_DataExploration_AfterPreprocessing(preprocessed_path_no_tfidf)
    
    #Feature selection
    run_FeatureSelection(preprocessed_path_no_tfidf)
    
    #Modeling_NaiveBayes
    
    KFold = int(input("Select KFold option for NaiveBayes Model K : "))
    run_NaiveBayes_TF_IDF(preprocessed_path_tfidf, KFold = 5)
    
    #Modeling_DecisionTrees
    KFold = int(input("Select KFold option for DecisionTree Model K : "))
    depth = int(input("Select Max Depth of Decision Tree : "))
    run_decisionTree(preprocessed_path_no_tfidf, KFold, depth)
    
    #Modeling RandomForsets
    KFold = int(input("Select KFold option for RandomForest Model K : "))
    depth = int(input("Select Max Depth of RandomForest : "))
    n_estimators = int(input("Select n_estimators of RandomForest : "))
    run_RandomForest(preprocessed_path_no_tfidf, KFold, depth, n_estimators)
    
    #Modeling KMeansClustering
    k = int(input("Select K for KMeans Cluster Model : "))
    run_SpamCluster(label_messsage_path, k)
    
    #Evaluation RandomForests with SMOTE
    SMOTE : str
    SMOTE = input("Select Using SMOTE (Yes / NO) : ").strip().lower()

    if  (SMOTE.lower() == 'yes') :
        KFold = int(input("Select KFold option for RandomForest Model K : "))
        run_RandomForest_SMOTE(preprocessed_path_no_tfidf, KFold)
    elif (SMOTE.lower() == 'no') :
        #Evaluation RandomForests without SMOTE
        KFold = int(input("Select KFold option for RandomForest Model K : "))
        run_RandomForest_Eval(preprocessed_path_no_tfidf, KFold = 5)
        
    if SMOTE not in ['yes', 'no']:
        print("Invalid Value entered")
        exit(1)
    
    #Evaluation KMeansClustering
    k = int(input("Select K for KMeans Cluster Model : "))
    run_KMeans_Eval(label_messsage_path, k)

if __name__ == "__main__":
    main()
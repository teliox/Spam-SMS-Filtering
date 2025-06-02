# Spam-SMS-Filtering
DataScience Term Project

***

## Objective Setting
Classification Model that automatically filters Spam SMS due to data leakage

***

## Process
Data Exploration

Data Preprocessing

Data Analysis

Modeling

Evaluation and Analysis

***

## Architecture

![스크린샷 2025-05-31 오전 7 32 25](https://github.com/user-attachments/assets/657d9c5d-7ba4-4522-9b72-1f84b6a52022)

***

## ! Best(and Top 5) Parameter Combination(For Standard Scaling with LabelEncoding, RandomForest Model) !

<img width="1017" alt="스크린샷 2025-06-02 오후 7 29 43" src="https://github.com/user-attachments/assets/a309a99b-47be-479a-99af-69080d570329" />

***

DataExploration

	DataExploration.py : Data Exploration before preprocessing

	DataExploration_AfterPreprocessing.py : Data Exploration (Distiribution) after preprocessing
 
	df_no_tfidf.csv : preprocessed dataset
  
	spam.csv : raw dataset

DataPreprocessing

	Preprocessing.py
 
	FeatureSelection.py

	df_no_tfidf.csv : preprocessed data without TF-IDF features

	df_with_tfidf.csv : preprocessed data with TF-IDF

	spam.csv

Modeling

Classification

	decicionTree.py 

	RandomForest.py

	NaiveBayes_TF-IDF.py

	df_no_tfidf.csv

	df_with_tfidf.csv
	
kmeansClustering
	SpamCluster.py
 
	df_label_message_only.csv
 
Evaluation
	Classification
 
	RandomForest_SMOTE.py : RandomForest model with SMOTE
 
	RandomForest.py : RandomForest evaluation with ROC, AUC, Randomized Grid Search (without SMOTE) kmeansClustering
 
	KMeans_Eval.py : KMeans Clustering model evaluation with PCA explained ratio, Elbow method
 
	df_label_message_only.csv : raw dataset
 
***
### EndToEnd.py

<img width="870" alt="스크린샷 2025-06-02 오후 7 35 36" src="https://github.com/user-attachments/assets/922762ec-a119-439e-ac01-85566d217dd4" />

<img width="870" alt="스크린샷 2025-06-02 오후 7 36 02" src="https://github.com/user-attachments/assets/658df60d-43f5-4a0d-aaad-63b57a493885" />

<img width="870" alt="스크린샷 2025-06-02 오후 7 36 18" src="https://github.com/user-attachments/assets/1c18cd42-1685-4398-a3e4-a451a461e851" />

You can Select Parameters

Scaling method (Standard / Robust)

TF-IDF Feature Dimension (150 / 300 / 450 / 600)

Every Model 's KFold K

Tree's Max Depth

Random Forest's n_estimators

Using SMOTE or not







***
## Members
202135752 김태량

202135750 김지현

202135814 이정균

202135840 조우현

202037006 권도윤

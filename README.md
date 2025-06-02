# Spam-SMS-Filtering
DataScience Term Project

***

## Objective Setting
Troubleshooting problems caused by personal information leakage caused by personal information leakage damage

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
### DataExploration.py
<img width="384" alt="image" src="https://github.com/user-attachments/assets/7b3bbab6-78d4-412f-8957-5df83115ed16" />

Load Data, Labeling, Distribution
<img width="388" alt="image" src="https://github.com/user-attachments/assets/322f8250-e524-4f2d-ac66-e0e0832e2d31" />

Identify characters that are over the ASCII code range and output what characters are, count how many they are

<img width="383" alt="image" src="https://github.com/user-attachments/assets/514e6359-73bc-462f-9834-c9d576f2df95" />


<img width="383" alt="image" src="https://github.com/user-attachments/assets/e04b363e-64d2-4848-a917-cf5e4806b200" />

Visualization & method referenced link

### DataExploration_AfterPreprocessing.py
<img width="384" alt="image" src="https://github.com/user-attachments/assets/1e8b7e7f-2855-4287-b397-9d9e69841154" />

Code to visualize statistics for numerical and categorical data after preprocessing


<img width="384" alt="image" src="https://github.com/user-attachments/assets/01a90e0b-5ddc-4095-b1fd-31902a0c2045" />

Plot Histogram for numerical feature











***
## Members
202135752 김태량
202135750 김지현
202135814 이정균
202135840 조우현
202037006 권도윤

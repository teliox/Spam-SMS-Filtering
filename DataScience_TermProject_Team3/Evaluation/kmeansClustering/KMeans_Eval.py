import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def run_KMeans_Eval(csv_path: str, k = 4):
    #Load the dataset
    df = pd.read_csv(csv_path)

    # Filter spam messages
    spam_df = df[df["label"] == 1].copy()

    print(f"Spam Messages Count: {len(spam_df)}")

    #Preprocess the messages
    #to Lowercase, Use Http -> <URL> www -> <URL> Token @ -> <EMAIL>
    def preprocess_text(text):
        text = text.lower()
        text = text.replace("http", " <URL> ").replace("www", " <URL> ")
        text = text.replace("@", " <EMAIL> ")
        return text

    spam_df["cleaned_message"] = spam_df["message"].astype(str).apply(preprocess_text)

    #TF-IDF vectorization
    #Max_df : Remove words that appear in more than 95% of all documents. (have little information)
    #Min_df : Use only words that appear in at least two documents. (Rare words can be noisy)
    #stop_words='english' : Automatic removal of English stopwords.
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X_tfidf = vectorizer.fit_transform(spam_df["cleaned_message"])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_tfidf)
    spam_df["cluster"] = cluster_labels

    # Silhouette Score
    sil_score = silhouette_score(X_tfidf, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    #Extracting Top keywords
    #Extract words with high TF-IDF mean values from statements belonging to the cluster (cluster_id)
    #Returns the main keyword (topN) that represents the cluster
    def get_top_keywords(cluster_id, top_n=8):
        idx = np.where(spam_df["cluster"] == cluster_id)[0]
        mean_tfidf = X_tfidf[idx].mean(axis=0).A1 #.A1 transforms sparse matrices into 1D NumPy arrays
        top_indices = mean_tfidf.argsort()[::-1][:top_n] #Extract top_n indexes in order of largest mean TF-IDF values
        return [vectorizer.get_feature_names_out()[i] for i in top_indices] #Return word names corresponding to the upper index to the list

    #Extracting Top messages
    #Select the message closest to the cluster center as "representative message" in each cluster
    def get_representative_message(cluster_id):
        idx = np.where(spam_df["cluster"] == cluster_id)[0]
        centroid = kmeans.cluster_centers_[cluster_id]
        dist = np.linalg.norm(X_tfidf[idx] - centroid, axis=1) #Calculate the L2 distance between each message vector and the center
        rep_idx = idx[dist.argmin()]
        return spam_df.iloc[rep_idx]["message"]

    #Name a Meaningful Cluster
    cluster_names = {}
    for i in range(k):
        keywords = get_top_keywords(i)
        if any(word in keywords for word in ["free", "win", "claim", "congratulations"]):
            cluster_names[i] = "Freebie / Prize Spam"
        elif any(word in keywords for word in ["loan", "cash", "urgent", "offer"]):
            cluster_names[i] = "Loan / Financial Spam"
        elif any(word in keywords for word in ["click", "account", "verify", "login", "identifier"]):
            cluster_names[i] = "Phishing / Account Access"
        elif any(word in keywords for word in ["adult", "sex", "hot", "meet"]):
            cluster_names[i] = "Adult Content Spam"
        else:
            cluster_names[i] = "Other / Mixed Type"

    #Printing Result
    print("\nCluster Analysis:")
    for i in range(k):
        print(f"\n Cluster {i} ({cluster_names[i]}):")
        print(" [Top Keywords] :", ', '.join(get_top_keywords(i)))
        print("\n [Representative Message] :", get_representative_message(i))

    #Visualization 2D PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    spam_df["pca_x"] = X_pca[:, 0]
    spam_df["pca_y"] = X_pca[:, 1]
    explained_var_ratio = pca.explained_variance_ratio_

    print(f"\nPCA Explained Variance Ratio (2D):")
    print(f" *PC1: {explained_var_ratio[0]:.4f}")
    print(f" *PC2: {explained_var_ratio[1]:.4f}")

    plt.figure(figsize=(10, 7))
    for i in range(k):
        subset = spam_df[spam_df["cluster"] == i]
        plt.scatter(subset["pca_x"], subset["pca_y"], label=cluster_names[i], alpha=0.6, s=50)

    plt.title("PCA Projection of Spam Messages (K-Means Clusters)", fontsize=14)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    from sklearn.metrics import pairwise_distances_argmin_min

    print("\nClustering with k=2, 3, 4 for comparison")

    for test_k in [2, 3, 4]:
        test_kmeans = KMeans(n_clusters=test_k, random_state=42, n_init=10)
        test_labels = test_kmeans.fit_predict(X_tfidf)
        sil_score_k = silhouette_score(X_tfidf, test_labels)
        print(f"\n k={test_k} | Silhouette Score = {sil_score_k:.4f}")
        
        #Similar method with function : get_top_keywords
        for cluster_id in range(test_k):
            cluster_idx = np.where(test_labels == cluster_id)[0]
            cluster_mean = X_tfidf[cluster_idx].mean(axis=0).A1
            top_words = np.array(vectorizer.get_feature_names_out())[cluster_mean.argsort()[::-1][:8]]
            closest, _ = pairwise_distances_argmin_min(test_kmeans.cluster_centers_[cluster_id].reshape(1, -1), X_tfidf[cluster_idx])
            rep_msg = spam_df.iloc[cluster_idx[closest[0]]]["message"]
            print(f" [Cluster {cluster_id}: Top Keywords] = {', '.join(top_words)}")
            print(f"\n [Representative Msg] : {rep_msg[:80]}...")

    #Elbow Method
    print("\nElbow Method")
    inertias = []
    K_range = range(2, k+1)
    for k_val in K_range:
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        km.fit(X_tfidf)
        inertias.append(km.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K_range, inertias, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Silhouette Score Plot
    print("\nSilhouette Scores by k")
    sil_scores = []
    for k_val in K_range:
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = km.fit_predict(X_tfidf)
        score = silhouette_score(X_tfidf, labels)
        sil_scores.append(score)
        print(f"\n k={k_val}: Silhouette Score = {score:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(K_range, sil_scores, marker='s', color='purple')
    plt.title("Silhouette Scores vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
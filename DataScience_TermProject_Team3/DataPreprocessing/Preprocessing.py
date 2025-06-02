import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def run_Preprocessing(csv_path: str, Scaler: str, max_features = 300):
    save_dir = os.path.dirname(csv_path)
    
    #NLTK resource download (stopwords)
    nltk.download('stopwords')

    #Load CSV file and store columns
    df = pd.read_csv(csv_path, encoding="latin-1")[['v1', 'v2']]
    df.columns = ['label', 'message']

    #Mapping lable : Ham -> 0, Spam -> 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    #Message to String
    df['message'] = df['message'].astype(str)

    #Delete Corrputed Message
    def contains_non_ascii(text):
        return any(ord(char) > 127 for char in text)
    df = df[~df['message'].apply(contains_non_ascii)].reset_index(drop=True)

    #Delete duplicated ham message that remain only 1 thing
    ham = df[df['label'] == 0]
    spam = df[df['label'] == 1]
    ham_cleaned = ham.drop_duplicates(subset='message', keep='first')
    df = pd.concat([ham_cleaned, spam], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    #Feature Creation
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
    df['char_length'] = df['message'].apply(lambda x: len(str(x)))
    df['digit_count'] = df['message'].apply(lambda x: len(re.findall(r'\d', str(x))))
    df['uppercase_count'] = df['message'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df['special_char_count'] = df['message'].apply(lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', str(x))))
    df['has_url'] = df['message'].str.contains(r'http[s]?://|www\.', regex=True).astype(int)

    #Spam keyword
    spam_keywords = ['free', 'win', 'cash', 'prize', 'claim', 'urgent']
    df['has_keywords'] = df['message'].apply(lambda x: int(any(kw in str(x).lower() for kw in spam_keywords)))
    df['starting_word'] = df['message'].apply(lambda x: str(x).split()[0].lower() if len(str(x).split()) > 0 else '')

    #Ratio Feature + keyword count
    df['digit_ratio'] = df['digit_count'] / (df['char_length'] + 1e-5)
    df['uppercase_ratio'] = df['uppercase_count'] / (df['char_length'] + 1e-5)

    def count_spam_keywords(text):
        text = text.lower()
        return sum(text.count(kw) for kw in spam_keywords)

    df['spam_keyword_count'] = df['message'].apply(count_spam_keywords)

    #Stemming (Clean Text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    df['clean_text'] = df['message'].str.lower()
    df['clean_text'] = df['clean_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

    #Catrgorical Feature
    df['length_bin'] = pd.cut(df['char_length'], bins=[-1, 50, 150, float('inf')], labels=['short', 'medium', 'long'])
    df['digit_bin'] = pd.cut(df['digit_count'], bins=[-1, 0, 3, 10, float('inf')], labels=['0', '1~3', '4~10', '11~'])

    #Encoding Catrgorical Feature
    le_length = LabelEncoder()
    df['length_bin_encoded'] = le_length.fit_transform(df['length_bin'].astype(str))

    le_digit = LabelEncoder()
    df['digit_bin_encoded'] = le_digit.fit_transform(df['digit_bin'].astype(str))

    top_words = df['starting_word'].value_counts().nlargest(10).index
    df['starting_word_cat'] = df['starting_word'].apply(lambda x: x if x in top_words else 'etc')
    le_starting = LabelEncoder()
    df['starting_word_encoded'] = le_starting.fit_transform(df['starting_word_cat'])


    #Scaling Numerical features
    num_cols = ['word_count', 'char_length', 'digit_count', 'uppercase_count',
                'special_char_count', 'digit_ratio', 'uppercase_ratio', 'spam_keyword_count']

    if (Scaler.lower() == 'standard') :
        print(f"Scaling method : {Scaler.upper()}")
        std_scaler = StandardScaler()
        df_std_scaled = pd.DataFrame(std_scaler.fit_transform(df[num_cols]),
                                    columns=[f'std_{col}' for col in num_cols])
        
        df = pd.concat([df, df_std_scaled], axis = 1)
        
    elif (Scaler.lower() == 'robust'):
        print(f"Scaling method : {Scaler.upper()}")
        robust_scaler = RobustScaler()
        df_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(df[num_cols]),
                                        columns=[f'robust_{col}' for col in num_cols])
        
        df = pd.concat([df, df_robust_scaled], axis = 1)

    # df = pd.concat([df, df_std_scaled, df_robust_scaled], axis=1)

    #TF-IDF Vectorization (max_features : number of features that will be vectorized)
    tfidf_vectorizer = TfidfVectorizer(max_features = max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=[f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()])
    tfidf_df.index = df.index 

    df_with_tfidf = pd.concat([df, tfidf_df], axis=1)

    #Storing
    df.to_csv(os.path.join(save_dir, "spam_no_tfidf.csv"), index=False)
    df_with_tfidf.to_csv(os.path.join(save_dir, "spam_with_tfidf.csv"), index=False)

    print("Success Preprocessing")


#Drop duplicate : https://www.geeksforgeeks.org/python-pandas-dataframe-drop_duplicates/
#Feature creation : https://medium.com/@kamig4u/a-comprehensive-guide-to-feature-engineering-for-machine-learning-in-python-b017274129fe
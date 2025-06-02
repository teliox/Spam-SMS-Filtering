import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def run_exploration(csv_path: str):
    #Load CSV file
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')

    #column 0 : Label(Ham or Spam) / column 1 : Message
    label_col = df.columns[0]
    message_col = df.columns[1]

    #Total number of dataset
    total_records = len(df)
    print(f"Total Record: {total_records}")

    #Ham, Spam Distribution : Counting each label
    label_counts = df[label_col].value_counts()
    label_ratios = df[label_col].value_counts(normalize=True)

    print("\n[Ham vs Spam Distribution]")
    for label in label_counts.index:
        print(f"{label}: {label_counts[label]}개 ({label_ratios[label]*100:.2f}%)")

    #corrupted messages : Detecting non-ASCII words
    def contains_non_ascii(val):
        try:
            return any(ord(char) > 127 for char in str(val))
        except:
            return False

    #If there exist at least one non-ASCII word, masking it to true
    str_cols = df.select_dtypes(include=['object'])
    non_ascii_mask = str_cols.applymap(contains_non_ascii).any(axis=1)
    num_non_ascii_rows = non_ascii_mask.sum()
    print(f"\nNumber of records that contain corrupted words: {num_non_ascii_rows}")

    #Type of corrupted word : Extracting non-ASCII words
    non_ascii_chars = Counter()
    for text in str_cols[non_ascii_mask].values.flatten():
        try:
            for char in str(text):
                if ord(char) > 127:
                    non_ascii_chars[char] += 1
        except:
            continue

    #Ham vs Spam Distribution
    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
    plt.title('Ham vs Spam Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    #Type of corrupted words
    plt.figure(figsize=(10, 5))
    chars, freqs = zip(*non_ascii_chars.most_common(20))
    bars = plt.bar(chars, freqs, color='orange')
    plt.title('Type of Corrupted word')
    plt.ylabel('Frequency')
    plt.xlabel('Character')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    #Total number of corrupted records
    plt.text(0.95, 0.95, f"Total corrupted rows: {num_non_ascii_rows}", ha='right', va='top',
            transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    plt.tight_layout()
    plt.show()

    #Message length distribution
    df["msg_length"] = df[message_col].astype(str).apply(len)
    plt.figure(figsize=(8, 4))
    plt.hist(df["msg_length"], bins=40, color='teal', edgecolor='black')
    plt.title('Message Lengths Histogram')
    plt.xlabel('Message Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Reference link : https://www.askpython.com/python/string/detect-ascii-characters-in-strings
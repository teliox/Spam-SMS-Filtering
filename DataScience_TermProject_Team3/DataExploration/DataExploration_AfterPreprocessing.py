import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_DataExploration_AfterPreprocessing(csv_path: str):
    #Load Data
    df = pd.read_csv(csv_path)
    y = df['label']

    print("\n Columns : ")
    print(df.columns.tolist())

    #Feature Definition
    numeric_features = [
        'word_count', 'char_length', 'digit_count', 'uppercase_count',
        'special_char_count', 'digit_ratio', 'uppercase_ratio', 'spam_keyword_count'
    ]

    #Define binary and categorical features
    binary_features = ['has_url', 'has_keywords']
    categorical_features = ['starting_word_encoded', 'length_bin_encoded', 'digit_bin_encoded']
    all_cat_features = binary_features + categorical_features

    #Plot PieChart: Spam : Ham Distribution
    label_counts = df['label'].value_counts()
    labels = ['Ham', 'Spam']
    sizes = [label_counts[0], label_counts[1]]
    colors = ['skyblue', 'salmon']
    counts_labels = [
        f'Ham\n{sizes[0]} ({sizes[0]/sum(sizes)*100:.1f}%)',
        f'Spam\n{sizes[1]} ({sizes[1]/sum(sizes)*100:.1f}%)'
    ]

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=counts_labels, startangle=90, colors=colors, textprops={'fontsize': 12})
    plt.title('Spam vs Ham Distribution')
    plt.axis('equal')
    plt.show()

    #Histogram of numerical features
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features):
        sns.histplot(data=df[df['label'] == 0], x=col, bins=40, kde=True, stat="density",
                    color='skyblue', label='Ham (0)', ax=axes[i], edgecolor='black')
        sns.histplot(data=df[df['label'] == 1], x=col, bins=40, kde=True, stat="density",
                    color='salmon', label='Spam (1)', ax=axes[i], edgecolor='black', alpha=0.6)
        
        axes[i].set_title(f'{col} Distribution (Normalized)', fontsize=11)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Density')
        axes[i].legend()

    # Remove empty axes
    for j in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    #Boxplot for detecting outliers
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 12))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features):
        sns.boxplot(data=df, x='label', y=col, hue='label', palette='Set2', ax=axes[i], legend=False)
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel('Label (0 = Ham, 1 = Spam)')
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()

    #Bar Chart for Categorical Features
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(all_cat_features):
        cat_plot = pd.crosstab(df[col], df['label'], normalize='index') * 100
        cat_plot.plot(kind='bar', stacked=True, ax=axes[i], colormap='Set2', edgecolor='black')
        axes[i].set_title(f'Distribution by {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Percentage')
        axes[i].legend(title='Label', labels=['Ham', 'Spam'])

    #Remove extra axes if any
    for j in range(len(all_cat_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

!pip install kaggle

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!unzip /content/Genetic.zip

df = pd.read_csv('/content/Medicine_Details.csv')

df.head()

df.tail()

df.describe()

df.info()

df.columns

df.isnull().sum()

df.duplicated().sum()


clean_df = df.drop_duplicates()

clean_df.head()

composition_value_counts = clean_df['Composition'].value_counts()
composition_value_counts

composition_names = composition_value_counts.index.tolist()
salts_name = composition_names[:30]
salts_name

import matplotlib.pyplot as plt

# Assuming 'salts_name' contains the top 30 composition names
# Replace 'salts_name' with your list of compositions

# Generate frequencies for each composition name
frequencies = [clean_df['Composition'].value_counts()[name] for name in salts_name]

# Create a bar plot with customizations
plt.figure(figsize=(10, 8))
plt.barh(salts_name, frequencies, color='skyblue', edgecolor='black')

plt.xlabel('Frequency')
plt.ylabel('Composition Names')
plt.title('Top 30 Composition Frequencies')
plt.gca().invert_yaxis()  # Invert y-axis to display the most frequent at the top

# Add frequency values as text on the bars
for i, v in enumerate(frequencies):
    plt.text(v + 0.2, i, str(v), color='black', va='center')

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()


side_effects_counts = clean_df['Side_effects'].value_counts()

# Create a dictionary to store medicines for each side effect
side_effects_medicines = {}

# Loop through unique side effects and collect associated medicines
for side_effect, count in side_effects_counts.items():
    # Filter DataFrame for each unique side effect
    medicines_for_side_effect = clean_df.loc[clean_df['Side_effects'] == side_effect, 'Medicine Name'].tolist()

    # Store the list of medicines for the side effect in the dictionary
    side_effects_medicines[side_effect] = medicines_for_side_effect

# Loop through the top 10 side effects and print their information
for idx, (side_effect, medicines) in enumerate(side_effects_medicines.items()):
    if idx >= 10:
        break  # Exit loop after printing 10 occurrences

    print(f"Side Effect: {side_effect}")
    print(f"Number of Occurrences: {side_effects_counts[side_effect]}")
    print(f"Medicines: {medicines}")
    print("------------------------")

# Get the value counts of uses
uses_counts = clean_df['Uses'].value_counts().head(10)

# Create a dictionary to store medicines for each use
uses_medicines = {}

# Loop through unique uses and collect associated medicines
for use, count in uses_counts.items():
    # Filter DataFrame for each unique use
    medicines_for_use = clean_df.loc[clean_df['Uses'] == use, 'Medicine Name'].tolist()

    # Store the list of medicines for the use in the dictionary
    uses_medicines[use] = medicines_for_use

# Display only the first 10 uses and their associated medicines
for idx, (use, medicines) in enumerate(uses_medicines.items()):
    if idx >= 10:
        break  # Exit loop after printing 10 occurrences

    print(f"Use: {use}")
    print(f"Number of Occurrences: {uses_counts[use]}")
    print(f"Medicines: {medicines}")
    print("------------------------")

import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'clean_df' contains the necessary data

# Get the value counts of side effects and limit to top 10
side_effects_counts = clean_df['Side_effects'].value_counts().head(10)

# Create a dictionary to store medicines for each side effect
side_effects_medicines = {}

# Loop through unique side effects and collect associated medicines
for side_effect in side_effects_counts.index:
    # Filter DataFrame for each unique side effect
    medicines_for_side_effect = clean_df.loc[clean_df['Side_effects'] == side_effect, 'Medicine Name'].tolist()

    # Store the list of medicines for the side effect in the dictionary
    side_effects_medicines[side_effect] = medicines_for_side_effect

# Create a bar chart of top 10 side effects and their counts with improved aesthetics
plt.figure(figsize=(12, 8))
bar_plot = side_effects_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Adding annotations for each bar with custom formatting
for i, count in enumerate(side_effects_counts):
    plt.text(i, count + 1, f'{count}', ha='center', va='bottom', fontsize=10)

# Draw a horizontal line for the mean count
mean_count = side_effects_counts.mean()
plt.axhline(mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.2f}')
plt.legend()

# Draw a horizontal line for the median count
median_count = side_effects_counts.median()
plt.axhline(median_count, color='green', linestyle="--", label=f'Median: {median_count:.2f}')
plt.legend()

plt.title('Top 10 Occurrences of Side Effects')
plt.xlabel('Side Effects')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

top_10_uses = uses_counts.head(10)

# Calculate median and mean values
median_value = top_10_uses.median()
mean_value = top_10_uses.mean()

# Create a bar chart for top 10 uses and their occurrences
plt.figure(figsize=(13, 10))
bar_plot = top_10_uses.plot(kind='bar', color='skyblue', edgecolor='black')

# Adding annotations for each bar
for i, count in enumerate(top_10_uses):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)

# Plotting median line with annotation
plt.axhline(median_value, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median_value:.2f}')
plt.text(len(top_10_uses) - 1, median_value, f'Median: {median_value:.2f}', color='red', va='bottom', ha='right', fontsize=10)

# Plotting mean line with annotation
plt.axhline(mean_value, color='green', linestyle='-.', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
plt.text(len(top_10_uses) - 1, mean_value, f'Mean: {mean_value:.2f}', color='green', va='bottom', ha='right', fontsize=10)

plt.title('Top 10 Occurrences of Uses and Associated Medicines', fontsize=14)
plt.xlabel('Uses', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, top_10_uses.max() * 1.1)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_uses = tfidf_vectorizer.fit_transform(clean_df['Uses'].astype(str))
tfidf_matrix_composition = tfidf_vectorizer.fit_transform(clean_df['Composition'].astype(str))
tfidf_matrix_side_effects = tfidf_vectorizer.fit_transform(clean_df['Side_effects'].astype(str))

# Ensure all matrices have the same number of rows
min_rows = min(tfidf_matrix_uses.shape[0], tfidf_matrix_composition.shape[0], tfidf_matrix_side_effects.shape[0])

# Trim matrices to have the same number of rows
tfidf_matrix_uses = tfidf_matrix_uses[:min_rows]
tfidf_matrix_composition = tfidf_matrix_composition[:min_rows]
tfidf_matrix_side_effects = tfidf_matrix_side_effects[:min_rows]

from scipy.sparse import hstack
# Combine the matrices horizontally
tfidf_matrix_combined = hstack((tfidf_matrix_uses, tfidf_matrix_composition, tfidf_matrix_side_effects))

tfidf_matrix_combined

cosine_sim_combined = cosine_similarity(tfidf_matrix_combined, tfidf_matrix_combined)
cosine_sim_combined

def recommend_medicines_by_usage(medicine_name, tfidf_matrix_uses, clean_df):
    # Get the index of the medicine
    medicine_index = clean_df[clean_df['Medicine Name'] == medicine_name].index[0]

    # Calculate cosine similarity between the given medicine and others based on usage
    sim_scores = cosine_similarity(tfidf_matrix_uses, tfidf_matrix_uses[medicine_index])

    # Get indices of top similar medicines (excluding the queried one)
    sim_scores = sim_scores.flatten()
    similar_indices = sim_scores.argsort()[::-1][1:6]  # Top 5 similar medicines

    # Get recommended medicine names
    recommended_medicines = clean_df.iloc[similar_indices]['Medicine Name'].tolist()

    return recommended_medicines

query = "Lobet 20mg Injection"
recommended_medicines = recommend_medicines_by_usage(query, tfidf_matrix_uses, clean_df)
print(recommended_medicines)

def recommend_medicines_by_symptoms(symptoms, tfidf_vectorizer, tfidf_matrix_uses, clean_df):
    # Create a string from the given symptoms
    symptom_str = ' '.join(symptoms)

    # Transform the symptom string using the TF-IDF vectorizer
    symptom_vector = tfidf_vectorizer.transform([symptom_str])

    # Calculate cosine similarity between the symptom vector and all medicine vectors
    sim_scores = cosine_similarity(tfidf_matrix_uses, symptom_vector)

    # Get indices of top similar medicines
    sim_scores = sim_scores.flatten()
    similar_indices = sim_scores.argsort()[::-1][:5]  # Top 5 similar medicines

    # Get recommended medicine names
    recommended_medicines = clean_df.iloc[similar_indices]['Medicine Name'].tolist()

    return recommended_medicines

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer for symptoms
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the 'Uses' column to create the TF-IDF matrix for symptoms
tfidf_matrix_uses = tfidf.fit_transform(clean_df['Uses'])

# Now, you can call the recommend_medicines_by_symptoms function
query = ["Diabetes"]  # Convert the single symptom to a list
recommended_medicines = recommend_medicines_by_symptoms(query, tfidf, tfidf_matrix_uses, clean_df)
print(recommended_medicines)

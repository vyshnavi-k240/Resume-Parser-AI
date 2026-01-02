# RESUME PARSER AI PROJECT

import pandas as pd
import nltk
import re
import string
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: SETUP AND DOWNLOADS

print("Step 1: Setting up NLTK...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab') # Specific download for newer Python versions

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("✅ Setup Complete.")

#CLEAN THE TEXT

def clean_text(text):
    
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# STEP 2: LOAD DATA

print("\nStep 2: Loading Dataset...")

# Name of your CSV file. 
csv_filename = 'UpdatedResumeDataSet.csv'

if not os.path.exists(csv_filename):
    print(f"❌ ERROR: I cannot find '{csv_filename}'.")
    print(f"   Please check if the file is in this folder: {os.getcwd()}")
    exit()

# Read the file
df = pd.read_csv(csv_filename)
print(f"✅ Dataset loaded. Found {len(df)} resumes.")

# Identify the column with the Resume Text 
resume_col = 'Resume'
if resume_col not in df.columns:
    resume_col = df.columns[1]
    print(f"⚠️ 'Resume' column not found. Using '{resume_col}' instead.")

# STEP 3: PREPROCESSING (CLEANING)

print("\nStep 3: Cleaning Resume Text...")
df['Cleaned_Resume'] = df[resume_col].apply(clean_text)
print("✅ Cleaning Complete.")

# STEP 4: MACHINE LEARNING (CLUSTERING)

print("\nStep 4: Training AI Model...")
vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
X = vectorizer.fit_transform(df['Cleaned_Resume'])

# B. Group similar resumes (Clustering)
# We will group them into 4 general categories
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_
print(f"✅ Model trained. Resumes grouped into {k} clusters.")

# STEP 5: VISUALIZATION (SAVE GRAPH)

print("\nStep 5: Creating Visualization...")

# Create 'output' folder if it doesn't exist
if not os.path.exists('../output'):
    os.makedirs('../output')

# Reduce data to 2 dimensions for plotting
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X.toarray())

# Draw the graph
plt.figure(figsize=(10, 7))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Resume Clusters Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Group')

# Save the graph instead of showing it (safer for scripts)
plt.savefig('../output/clusters_plot.png', dpi=150, bbox_inches='tight')
plt.close() 
print("✅ Graph saved to: output/clusters_plot.png")

# STEP 6: DYNAMIC CANDIDATE SCORING

print("\n" + "="*50)
print("Step 6: HR CANDIDATE SEARCH")
print("="*50)

# 1. Ask HR to type the Job Description
print("\nPlease type the Job Description below (Press 'Enter' when done):")
print("-" * 50)

# This line waits for the user to type and press Enter
user_jd = input(">> ")

# Check if user typed something (not empty)
if not user_jd.strip():
    print("❌ Error: No job description provided.")
else:
    print("\n👉 Analyzing your Job Description...")
    
    # 2. Clean the typed text (using our same function)
    cleaned_jd = clean_text(user_jd)

    # 3. Turn Job Description into numbers
    jd_vector = vectorizer.transform([cleaned_jd])

    # 4. Compare against all resumes
    similarity_scores = cosine_similarity(jd_vector, X)
    
    # 5. Add scores to dataframe
    df['Score'] = similarity_scores[0]
    df['Score_Out_of_10'] = df['Score'] * 10

    # 6. Get the Top 5 Matches
    top_candidates = df.sort_values(by='Score', ascending=False).head(5)

    # 7. Check for Eligibility
    # We set a threshold. If the BEST match is less than 1.5 out of 10, we say "No match".
    best_score = top_candidates.iloc[0]['Score_Out_of_10']
    threshold = 1.5 

    if best_score < threshold:
        print("\n❌ Result: No suitable candidates found for this job description.")
        print(f"(The closest match was only {best_score:.2f}/10)")
    else:
        print("\n✅ Result: Candidates Found!")
        print(f"(Top Match Score: {best_score:.2f}/10)")
        print("-" * 50)
        
        # Determine which column is the "Name/Category" column
        label_col = 'Category'
        if label_col not in top_candidates.columns:
            label_col = df.columns[0] 

        for index, row in top_candidates.iterrows():
            rank = list(top_candidates.index).index(index) + 1
            print(f"Rank: {rank}")
            print(f"{label_col}: {row[label_col]}")
            print(f"Match Score: {row['Score_Out_of_10']:.2f} / 10")
            print(f"Snippet: ...{row['Cleaned_Resume'][100:160]}...")
            print("-" * 50)

# FINAL SAVE (Still saves the data so you can review it)

output_path = '../output/clustered_resumes.csv'
df.to_csv(output_path, index=False)
print(f"\n📁 Data saved to: {output_path}")
print("🚀 Process Complete. Thank you for using Resume Parser AI.")
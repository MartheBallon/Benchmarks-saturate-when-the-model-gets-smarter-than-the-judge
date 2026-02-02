import sys
import os
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


df = pd.read_json("Omni-Math-2.jsonl", lines=True)
df_bad = df[df['tags'].apply(lambda x: len(x) > 0)]

# number of tagged questions
num_tagged_questions = df[df['tags'].apply(lambda x: len(x) > 0)].shape[0]
print(f"Number of tagged questions: {num_tagged_questions}")
print(f'Percentage of tagged questions: {num_tagged_questions / df.shape[0] * 100:.2f}%')

# number of added images
num_added_images = df['tags'].apply(lambda x: 1 if 'image' in x else 0).sum()
print(f"Number of added images: {num_added_images}")

# number of proofs 
num_proofs = df['tags'].apply(lambda x: 1 if 'proof' in x else 0).sum()
print(f"Number of proofs: {num_proofs}")

# number of estimation questions
num_estimation_questions = df['tags'].apply(lambda x: 1 if 'estimation' in x else 0).sum()
print(f"Number of estimation questions: {num_estimation_questions}")

# number of exluded questions
num_excluded_questions = df['tags'].apply(lambda x: 1 if 'should delete' in x else 0).sum()
print(f"Number of excluded questions: {num_excluded_questions}")

# number of edited questions
df_original = pd.read_json("Omni-MATH.jsonl", lines=True)
num_edited_questions = 0
for i, row in df.iterrows():
    original_problem = df_original.loc[df_original.index == row['id'] - 1, 'problem'].values[0]
    if row['problem'] != original_problem:
        num_edited_questions += 1
print(f"Number of edited questions: {num_edited_questions}")
print(f"Percentage of edited questions: {num_edited_questions / df.shape[0] * 100:.2f}%")
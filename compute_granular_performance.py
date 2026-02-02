import json
import sys
import os
from pathlib import Path
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import domain_performance, difficulty_performance

# --------------------------------------- GPT-5 --------------------------------------- #

# by omni-judge
gpt5_per_domain_by_oj = domain_performance('gpt5/gpt5_filtered_judged_by_oj.jsonl', "Omni-Math-2-Filtered.jsonl", "Omni-Judge")
gpt5_per_difficulty_by_oj = difficulty_performance("gpt5/gpt5_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")

nr_of_calculus_problems = gpt5_per_domain_by_oj['Calculus']['total']
nr_of_algebra_problems = gpt5_per_domain_by_oj['Algebra']['total']
# by gpt5mini
gpt5_per_difficulty_by_g5 = difficulty_performance("gpt5/gpt5_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
gpt5_per_domain_by_g5 = domain_performance("gpt5/gpt5_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")

nr_of_tier4_problems = gpt5_per_difficulty_by_g5['Tier 4']['total']

print(f"Number of Calculus problems: {nr_of_calculus_problems}")
print(f"Number of Algebra problems: {nr_of_algebra_problems}")
print(f"Number of Tier 4 problems: {nr_of_tier4_problems}")

# --------------------------------------- Claude --------------------------------------- #
# by omni-judge
claude_per_domain_by_oj = domain_performance("claude/claude_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")
claude_per_difficulty_by_oj = difficulty_performance("claude/claude_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")

# by gpt5mini
claude_per_difficulty_by_g5 = difficulty_performance("claude/claude_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
claude_per_domain_by_g5 = domain_performance("claude/claude_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
# --------------------------------------- DeepSeek --------------------------------------- #
# by omni-judge
deepseek_per_domain_by_oj = domain_performance("deepseek/deepseek_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")
deepseek_per_difficulty_by_oj = difficulty_performance("deepseek/deepseek_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")

# by gpt5mini
deepseek_per_difficulty_by_g5 = difficulty_performance("deepseek/deepseek_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
deepseek_per_domain_by_g5 = domain_performance("deepseek/deepseek_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
# --------------------------------------- Kimi --------------------------------------- #
# by omni-judge
kimi_per_domain_by_oj = domain_performance("kimi/kimi_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")
kimi_per_difficulty_by_oj = difficulty_performance("kimi/kimi_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")

# by gpt5mini
kimi_per_difficulty_by_g5 = difficulty_performance("kimi/kimi_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
kimi_per_domain_by_g5 = domain_performance("kimi/kimi_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
# --------------------------------------- Gemini --------------------------------------- #
# by omni-judge
gemini_per_domain_by_oj = domain_performance("gemini/gemini_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")
gemini_per_difficulty_by_oj = difficulty_performance("gemini/gemini_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl", "Omni-Judge")

# by gpt5mini
gemini_per_difficulty_by_g5 = difficulty_performance("gemini/gemini_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")
gemini_per_domain_by_g5 = domain_performance("gemini/gemini_filtered_judged_by_gpt5mini.jsonl", "Omni-Math-2-Filtered.jsonl", "gpt5mini")


# Dataframe with columns model, judge difference, domain, difficulty
DOMAINS = ['Algebra', 'Calculus', 'Geometry', 'Number Theory', 'Discrete Mathematics', 'Applied Mathematics']
DIFFICULTY_LEVELS = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']    

# One df with model, domain, performance difference columns
records_domain = []
for model_name, per_domain_by_oj, per_domain_by_g5 in [
    ('gpt5', gpt5_per_domain_by_oj, gpt5_per_domain_by_g5),
    ('claude', claude_per_domain_by_oj, claude_per_domain_by_g5),
    ('deepseek', deepseek_per_domain_by_oj, deepseek_per_domain_by_g5),
    ('kimi', kimi_per_domain_by_oj, kimi_per_domain_by_g5),
    ('gemini', gemini_per_domain_by_oj, gemini_per_domain_by_g5),
]:
    for domain in DOMAINS:
        record_oj = {
            'model': model_name,
            'judge': 'Omni-Judge',
            'domain': domain,
            'accuracy': per_domain_by_oj.get(domain, float('nan'))['accuracy'],
        }
        records_domain.append(record_oj)

        df_domain_omni_judge = pd.DataFrame(records_domain)
        
        record_g5 = {
            'model': model_name,
            'judge': 'gpt5mini',
            'domain': domain,
            'accuracy': per_domain_by_g5.get(domain, float('nan'))['accuracy'],
        }
        records_domain.append(record_g5)

        df_domain_gpt5mini = pd.DataFrame(records_domain)

# join two dfs and create new column with absolute difference in accuracy
df_domain_combined = pd.concat([df_domain_omni_judge, df_domain_gpt5mini])
df_domain_pivot = df_domain_combined.pivot_table(index=['model', 'domain'], columns='judge', values='accuracy').reset_index()
df_domain_pivot['accuracy_difference'] = df_domain_pivot['Omni-Judge'] - df_domain_pivot['gpt5mini']


# One df with model, difficulty, judge difference
records = []
for model_name, per_difficulty_by_oj, per_difficulty_by_g5 in [
    ('gpt5', gpt5_per_difficulty_by_oj, gpt5_per_difficulty_by_g5),
    ('claude', claude_per_difficulty_by_oj, claude_per_difficulty_by_g5),
    ('deepseek', deepseek_per_difficulty_by_oj, deepseek_per_difficulty_by_g5),
    ('kimi', kimi_per_difficulty_by_oj, kimi_per_difficulty_by_g5),
    ('gemini', gemini_per_difficulty_by_oj, gemini_per_difficulty_by_g5),
]:
    for difficulty in DIFFICULTY_LEVELS:
        record_oj = {
            'model': model_name,
            'judge': 'Omni-Judge',
            'difficulty': difficulty,
            'accuracy': per_difficulty_by_oj.get(difficulty, float('nan'))['accuracy'],
        }
        records.append(record_oj)

        df_difficulty_omni_judge = pd.DataFrame(records)
        
        record_g5 = {
            'model': model_name,
            'judge': 'gpt5mini',
            'difficulty': difficulty,
            'accuracy': per_difficulty_by_g5.get(difficulty, float('nan'))['accuracy'],
        }
        records.append(record_g5)

        df_difficulty_gpt5mini = pd.DataFrame(records)  
# join two dfs and create new column with absolute difference in accuracy
df_difficulty_combined = pd.concat([df_difficulty_omni_judge, df_difficulty_gpt5mini])
df_difficulty_pivot = df_difficulty_combined.pivot_table(index=['model', 'difficulty'], columns='judge', values='accuracy').reset_index()
df_difficulty_pivot['accuracy_difference'] = df_difficulty_pivot['Omni-Judge'] - df_difficulty_pivot['gpt5mini']

def compute_granular_performance():
    return df_domain_pivot, df_difficulty_pivot
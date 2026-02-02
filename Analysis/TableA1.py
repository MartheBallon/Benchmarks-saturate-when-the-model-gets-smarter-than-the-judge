import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import bayes_evals as be
from utils import get_dataframe_reasoning_models, get_dataframe_gpt5mini


omni_math_2 = pd.read_json("Omni-Math-2-Filtered.jsonl", lines=True)

# --------------------------------------- GPT-5 --------------------------------------- #
df_gpt5_judged_by_gpt5mini = get_dataframe_gpt5mini("gpt5/gpt5_filtered_judged_by_gpt5mini.jsonl")
df_gpt5_merged = omni_math_2.merge(df_gpt5_judged_by_gpt5mini, on='id')
df_gpt5_judged_by_oj = get_dataframe_reasoning_models("gpt5/gpt5_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")

performance_by_gpt5mini = df_gpt5_merged['judged_correct'].mean()
performance_by_oj = df_gpt5_judged_by_oj['score'].mean()

print(f"GPT-5 accuracy judged by GPT-5 mini: {performance_by_gpt5mini:.2%}")
print(f"GPT-5 accuracy judged by Omni-Judge: {performance_by_oj:.2%}")


# --------------------------------------- Claude --------------------------------------- #
df_claude_judged_by_gpt5mini = get_dataframe_gpt5mini("claude/claude_filtered_judged_by_gpt5mini.jsonl")
df_claude_merged = omni_math_2.merge(df_claude_judged_by_gpt5mini, on='id')
df_claude_judged_by_oj = get_dataframe_reasoning_models("claude/claude_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")

performance_claude_by_gpt5mini = df_claude_merged['judged_correct'].mean()
performance_claude_by_oj = df_claude_judged_by_oj['score'].mean()

print(f"Claude Sonnet 4.5 accuracy judged by GPT-5 mini: {performance_claude_by_gpt5mini:.2%}")
print(f"Claude Sonnet 4.5 accuracy judged by Omni-Judge: {performance_claude_by_oj:.2%}")


# --------------------------------------- DeepSeek --------------------------------------- #
df_deepseek_judged_by_gpt5mini = get_dataframe_gpt5mini("deepseek/deepseek_filtered_judged_by_gpt5mini.jsonl")
df_deepseek_merged = omni_math_2.merge(df_deepseek_judged_by_gpt5mini, on='id')
df_deepseek_judged_by_oj = get_dataframe_reasoning_models("deepseek/deepseek_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")

performance_deepseek_by_gpt5mini = df_deepseek_merged['judged_correct'].mean()
performance_deepseek_by_oj = df_deepseek_judged_by_oj['score'].mean()

print(f"DeepSeek v3.2 accuracy judged by GPT-5 mini: {performance_deepseek_by_gpt5mini:.2%}")
print(f"DeepSeek v3.2 accuracy judged by Omni-Judge: {performance_deepseek_by_oj:.2%}")


# ------------------------------------------ Kimi ------------------------------------------ #
df_kimi_judged_by_gpt5mini = get_dataframe_gpt5mini("kimi/kimi_filtered_judged_by_gpt5mini.jsonl")
df_kimi_merged = omni_math_2.merge(df_kimi_judged_by_gpt5mini, on='id')
df_kimi_judged_by_oj = get_dataframe_reasoning_models("kimi/kimi_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")

performance_kimi_by_gpt5mini = df_kimi_merged['judged_correct'].mean()
performance_kimi_by_oj = df_kimi_judged_by_oj['score'].mean()

print(f"Kimi K2 Thinking accuracy judged by GPT-5 mini: {performance_kimi_by_gpt5mini:.2%}")
print(f"Kimi K2 Thinking accuracy judged by Omni-Judge: {performance_kimi_by_oj:.2%}")

# ------------------------------------------ Gemini ------------------------------------------ #  
df_gemini_judged_by_gpt5mini = get_dataframe_gpt5mini("gemini/gemini_filtered_judged_by_gpt5mini.jsonl")
df_gemini_merged = omni_math_2.merge(df_gemini_judged_by_gpt5mini, on='id')
df_gemini_judged_by_oj = get_dataframe_reasoning_models("gemini/gemini_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")

performance_gemini_by_gpt5mini = df_gemini_merged['judged_correct'].mean()
performance_gemini_by_oj = df_gemini_judged_by_oj['score'].mean()

print(f"Gemini 3 Pro accuracy judged by GPT-5 mini: {performance_gemini_by_gpt5mini:.2%}")
print(f"Gemini 3 Pro accuracy judged by Omni-Judge: {performance_gemini_by_oj:.2%}")

# ------------------------------------------ Confidence intervals ------------------------------------------ #
records = [
    {
        'gpt5': score_gpt5,
        'claude': score_claude,
        'deepseek': score_deepseek,
        'kimi': score_kimi,
        'gemini': score_gemini
    }
    for score_gpt5, score_claude, score_deepseek, score_kimi, score_gemini in zip(df_gpt5_judged_by_oj['score'], df_claude_judged_by_oj['score'], df_deepseek_judged_by_oj['score'], df_kimi_judged_by_oj['score'], df_gemini_judged_by_oj['score'])
]
df_total_oj = pd.DataFrame(records)

records = [
    {
        'gpt5': score_gpt5,
        'claude': score_claude,
        'deepseek': score_deepseek,
        'kimi': score_kimi,
        'gemini': score_gemini
    }
    for score_gpt5, score_claude, score_deepseek, score_kimi, score_gemini in zip(df_gpt5_judged_by_gpt5mini['judged_correct'], df_claude_judged_by_gpt5mini['judged_correct'], df_deepseek_judged_by_gpt5mini['judged_correct'], df_kimi_judged_by_gpt5mini['judged_correct'], df_gemini_judged_by_gpt5mini['judged_correct'])

]
df_total_gpt5mini = pd.DataFrame(records)

records_merged = [
    {
        'gpt5': score_gpt5,
        'gpt5_2': score_gpt5_2,
        'claude': score_claude,
        'claude_2': score_claude_2,
        'deepseek': score_deepseek,
        'deepseek_2': score_deepseek_2,
        'kimi': score_kimi,
        'kimi_2': score_kimi_2,
        'gemini': score_gemini,
        'gemini_2': score_gemini_2
    }
    for score_gpt5, score_gpt5_2, score_claude, score_claude_2, score_deepseek, score_deepseek_2, score_kimi, score_kimi_2, score_gemini, score_gemini_2 in zip(df_gpt5_judged_by_oj['score'], df_gpt5_judged_by_gpt5mini['judged_correct'], df_claude_judged_by_oj['score'], df_claude_judged_by_gpt5mini['judged_correct'], df_deepseek_judged_by_oj['score'], df_deepseek_judged_by_gpt5mini['judged_correct'], df_kimi_judged_by_oj['score'], df_kimi_judged_by_gpt5mini['judged_correct'], df_gemini_judged_by_oj['score'], df_gemini_judged_by_gpt5mini['judged_correct'])
]


# Load the data (should NOT contain an index column)
eval_data = df_total_oj
eval_data_2 = df_total_gpt5mini

# Get the results either for individual LLMs (each column in the data)
# with a specified confidence level alpha (default=0.05)...
indep_intervals = be.independent_intervals(eval_data, alpha=0.05)
indep_intervals_2 = be.independent_intervals(eval_data_2, alpha=0.05)

print("CIs for Omni-Judge: ", indep_intervals)
print("CIs for GPT-5 mini: ", indep_intervals_2)
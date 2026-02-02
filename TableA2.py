import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pandas as pd

from utils import get_dataframe_reasoning_models, get_dataframe_gpt5mini

# --------------------------------------- GPT-5 --------------------------------------- #
# Merge with omni_math_2 to get full data
df_gpt5_judged_by_gpt5mini = get_dataframe_gpt5mini("gpt5/gpt5_filtered_judged_by_gpt5mini.jsonl")
df_gpt5_judged_by_oj = get_dataframe_reasoning_models("gpt5/gpt5_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")
df_gpt5_merged = df_gpt5_judged_by_gpt5mini.merge(df_gpt5_judged_by_oj, on='id')
disagreement_filtered_gpt5 = df_gpt5_judged_by_gpt5mini[df_gpt5_judged_by_gpt5mini['judged_correct'] != df_gpt5_judged_by_oj['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on gpt5 responses: {disagreement_filtered_gpt5}")

# Same for bad questions
df_gpt5_tagged_judged_by_omni_judge = get_dataframe_reasoning_models("gpt5/gpt5_tagged_judged_by_oj.jsonl", "Omni-Math-2-Tagged.jsonl")
df_gpt5_tagged_judged_by_gpt5mini = get_dataframe_gpt5mini("gpt5/gpt5_tagged_judged_by_gpt5mini.jsonl")
df_gpt5_merged_tagged = df_gpt5_tagged_judged_by_gpt5mini.merge(df_gpt5_tagged_judged_by_omni_judge, on='id')
disagreement_tagged_gpt5 = df_gpt5_merged_tagged[df_gpt5_merged_tagged['judged_correct'] != df_gpt5_merged_tagged['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on gpt5 BAD responses: {disagreement_tagged_gpt5}") 


# --------------------------------------- Claude --------------------------------------- #
df_claude_judged_by_gpt5mini = get_dataframe_gpt5mini("claude/claude_filtered_judged_by_gpt5mini.jsonl")
df_claude_judged_by_oj = get_dataframe_reasoning_models("claude/claude_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")
df_claude_merged = df_claude_judged_by_gpt5mini.merge(df_claude_judged_by_oj, on='id') # have to merge because omni-judge has 3 empty entries so df sizes differ
disagreement_filtered_claude = df_claude_merged[df_claude_merged['judged_correct'] != df_claude_merged['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on Claude responses: {disagreement_filtered_claude}")

df_claude_tagged_judged_by_oj = get_dataframe_reasoning_models("claude/claude_tagged_judged_by_oj.jsonl", "Omni-Math-2-Tagged.jsonl")
df_claude_tagged_judged_by_gpt5mini = get_dataframe_gpt5mini("claude/claude_tagged_judged_by_gpt5mini.jsonl")
df_claude_merged_tagged = df_claude_tagged_judged_by_gpt5mini.merge(df_claude_tagged_judged_by_oj, on='id')
disagreement_tagged_claude = df_claude_merged_tagged[df_claude_merged_tagged['judged_correct'] != df_claude_merged_tagged['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on Claude BAD responses: {disagreement_tagged_claude}")


# --------------------------------------- DeepSeek --------------------------------------- #
df_deepseek_judged_by_gpt5mini = get_dataframe_gpt5mini("deepseek/deepseek_filtered_judged_by_gpt5mini.jsonl")
df_deepseek_judged_by_oj = get_dataframe_reasoning_models("deepseek/deepseek_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")
df_deepseek_merged = df_deepseek_judged_by_gpt5mini.merge(df_deepseek_judged_by_oj, on='id') # have to merge because omni-judge has 3 empty entries so df sizes differ
disagreement_filtered_deepseek = df_deepseek_merged[df_deepseek_merged['judged_correct'] != df_deepseek_merged['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on DeepSeek responses: {disagreement_filtered_deepseek}")

df_deepseek_tagged_judged_by_oj = get_dataframe_reasoning_models("deepseek/deepseek_tagged_judged_by_oj.jsonl", "Omni-Math-2-Tagged.jsonl")
df_deepseek_tagged_judged_by_gpt5mini = get_dataframe_gpt5mini("deepseek/deepseek_tagged_judged_by_gpt5mini.jsonl")
df_deepseek_merged_tagged = df_deepseek_tagged_judged_by_gpt5mini.merge(df_deepseek_tagged_judged_by_oj, on='id')
disagreement_deepseek_tagged = df_deepseek_merged_tagged[df_deepseek_merged_tagged['judged_correct'] != df_deepseek_merged_tagged['score']].shape[0] # 1 empty entry here too

print(f"Number of disagreements between gpt5mini and omni-judge on DeepSeek BAD responses: {disagreement_deepseek_tagged}")
# ------------------------------------------ Kimi ------------------------------------------ #
df_kimi_judged_by_gpt5mini = get_dataframe_gpt5mini("kimi/kimi_filtered_judged_by_gpt5mini.jsonl")
df_kimi_judged_by_oj = get_dataframe_reasoning_models("kimi/kimi_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")
df_kimi_merged = df_kimi_judged_by_gpt5mini.merge(df_kimi_judged_by_oj, on='id') # have to merge because omni-judge has 3 empty entries so df sizes differ
disagreement_filtered_kimi = df_kimi_merged[df_kimi_merged['judged_correct'] != df_kimi_merged['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on Kimi responses: {disagreement_filtered_kimi}")

df_kimi_tagged_judged_by_oj = get_dataframe_reasoning_models("kimi/kimi_tagged_judged_by_oj.jsonl", "Omni-Math-2-Tagged.jsonl")
df_kimi_tagged_judged_by_gpt5mini = get_dataframe_gpt5mini("kimi/kimi_tagged_judged_by_gpt5mini.jsonl")
df_kimi_merged_tagged = df_kimi_tagged_judged_by_gpt5mini.merge(df_kimi_tagged_judged_by_oj, on='id') # 2 empty entries here too
disagreement_kimi_tagged = df_kimi_merged_tagged[df_kimi_merged_tagged['judged_correct'] != df_kimi_merged_tagged['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on Kimi BAD responses: {disagreement_kimi_tagged}")


# ------------------------------------------ Gemini ------------------------------------------ #  
df_gemini_judged_by_gpt5mini = get_dataframe_gpt5mini("gemini/gemini_filtered_judged_by_gpt5mini.jsonl")
df_gemini_judged_by_oj = get_dataframe_reasoning_models("gemini/gemini_filtered_judged_by_oj.jsonl", "Omni-Math-2-Filtered.jsonl")
df_gemini_merged = df_gemini_judged_by_gpt5mini.merge(df_gemini_judged_by_oj, on='id')
disagreement_clean_gemini = df_gemini_merged[df_gemini_merged['judged_correct'] != df_gemini_merged['score']].shape[0]
print(f"Number of disagreements between gpt5mini and omni-judge on Gemini responses: {disagreement_clean_gemini}")

df_gemini_tagged_judged_by_oj = get_dataframe_reasoning_models("gemini/gemini_tagged_judged_by_oj.jsonl", "Omni-Math-2-Tagged.jsonl")
df_gemini_tagged_judged_by_gpt5mini = get_dataframe_gpt5mini("gemini/gemini_tagged_judged_by_gpt5mini.jsonl")
df_gemini_merged_tagged = df_gemini_tagged_judged_by_gpt5mini.merge(df_gemini_tagged_judged_by_oj, on='id') # 2 empty entries here too
disagreement_gemini_tagged = df_gemini_merged_tagged[df_gemini_merged_tagged['judged_correct'] != df_gemini_merged_tagged['score']].shape[0]    
print(f"Number of disagreements between gpt5mini and omni-judge on Gemini BAD responses: {disagreement_gemini_tagged}")


# ------------------------------------------ Create excel files for manual annotation ------------------------------------------ #
# Disagreements on Omni-Math-2-Filtered
# GPT-5
size = 100
df_gpt5_merged = df_gpt5_merged.rename(columns={
    'judged_correct': 'score_gpt5mini',
    'score': 'score_omni_judge',
    'extracted_final_answer': 'final_answer_by_gpt5mini',
    'final_answer': 'final_answer_by_omni_judge',
    'answer': 'reference_answer',
    'Equivalence Judgement': 'justification_by_gpt5mini',
    'Justification': 'justification_by_omni_judge',   
})
#df_gpt5_merged = df_gpt5_merged.drop(columns=['correctness'])
df_gpt5_merged = df_gpt5_merged[[
    'id',
    'difficulty',
    'domain',
    'problem',
    'reference_answer',
    'final_answer_by_omni_judge',
    'justification_by_omni_judge',
    'score_omni_judge',
    'final_answer_by_gpt5mini',
    'justification_by_gpt5mini',
    'score_gpt5mini'
]]
disagreement_filtered_gpt5 = df_gpt5_merged[df_gpt5_merged['score_omni_judge'] != df_gpt5_merged['score_gpt5mini']]
sample_filtered_gpt5 = disagreement_filtered_gpt5.sample(n=size, random_state=42)
#sample_filtered_gpt5.to_excel('gpt5_filtered_disagreements_to_annotate.xlsx', index=False, engine='openpyxl')


# Disagreements on Omni-Math-2-Tagged
# GPT-5
df_gpt5_merged_tagged = df_gpt5_merged_tagged.rename(columns={
    'judged_correct': 'score_gpt5mini',
    'score': 'score_omni_judge',
    'extracted_final_answer': 'final_answer_by_gpt5mini',
    'final_answer': 'final_answer_by_omni_judge',
    'answer': 'reference_answer',
    'Equivalence Judgement': 'justification_by_gpt5mini',
    'Justification': 'justification_by_omni_judge',   
})
#df_gpt5_merged_tagged = df_gpt5_merged_tagged.drop(columns=['correctness'])
df_gpt5_merged_tagged = df_gpt5_merged_tagged[[
    'id',
    'difficulty',
    'domain',
    'problem',
    'reference_answer',
    'final_answer_by_omni_judge',
    'justification_by_omni_judge',
    'score_omni_judge',
    'final_answer_by_gpt5mini',
    'justification_by_gpt5mini',
    'score_gpt5mini'
]]
df_gpt5_merged_tagged = df_gpt5_merged_tagged[df_gpt5_merged_tagged['score_omni_judge'] != df_gpt5_merged_tagged['score_gpt5mini']]
#df_gpt5_merged_tagged.to_excel('gpt5_tagged_disagreements_to_annotate.xlsx', index=False, engine='openpyxl')

# claude
df_claude_merged_tagged = df_claude_merged_tagged.rename(columns={
    'judged_correct': 'score_gpt5mini',
    'score': 'score_omni_judge',
    'extracted_final_answer': 'final_answer_by_gpt5mini',
    'final_answer': 'final_answer_by_omni_judge',
    'answer': 'reference_answer',
    'Equivalence Judgement': 'justification_by_gpt5mini',
    'Justification': 'justification_by_omni_judge',   
})
#df_claude_merged_tagged = df_claude_merged_tagged.drop(columns=['correctness'])
df_claude_merged_tagged = df_claude_merged_tagged[[
    'id',
    'difficulty',
    'domain',
    'problem',
    'reference_answer',
    'final_answer_by_omni_judge',
    'justification_by_omni_judge',
    'score_omni_judge',
    'final_answer_by_gpt5mini',
    'justification_by_gpt5mini',
    'score_gpt5mini'
]]
df_claude_merged_tagged = df_claude_merged_tagged[df_claude_merged_tagged['score_omni_judge'] != df_claude_merged_tagged['score_gpt5mini']]
#df_claude_merged_tagged.to_excel('claude_tagged_disagreements_to_annotate.xlsx', index=False, engine='openpyxl')

# Deepseek
df_deepseek_merged_tagged = df_deepseek_merged_tagged.rename(columns={
    'judged_correct': 'score_gpt5mini',
    'score': 'score_omni_judge',
    'extracted_final_answer': 'final_answer_by_gpt5mini',
    'final_answer': 'final_answer_by_omni_judge',
    'answer': 'reference_answer',
    'Equivalence Judgement': 'justification_by_gpt5mini',
    'Justification': 'justification_by_omni_judge',
})

# Delete correctness columns
#df_deepseek_merged_tagged = df_deepseek_merged_tagged.drop(columns=['correctness'])

# Reorder columns
df_deepseek_merged_tagged = df_deepseek_merged_tagged[[
    'id',
    'difficulty',
    'domain',
    'problem',
    'reference_answer',
    'final_answer_by_omni_judge',
    'justification_by_omni_judge',
    'score_omni_judge',
    'final_answer_by_gpt5mini',
    'justification_by_gpt5mini',
    'score_gpt5mini'
]]

# Keep only the rows where the two judges disagree
df_deepseek_merged_tagged = df_deepseek_merged_tagged[df_deepseek_merged_tagged['score_omni_judge'] != df_deepseek_merged_tagged['score_gpt5mini']]
#df_deepseek_merged_tagged.to_excel('deepseek_tagged_disagreements_to_annotate.xlsx', index=False, engine='openpyxl')

# Kimi  
df_kimi_merged_tagged = df_kimi_merged_tagged.rename(columns={
    'judged_correct': 'score_gpt5mini',
    'score': 'score_omni_judge',
    'extracted_final_answer': 'final_answer_by_gpt5mini',
    'final_answer': 'final_answer_by_omni_judge',
    'answer': 'reference_answer',
    'Equivalence Judgement': 'justification_by_gpt5mini',
    'Justification': 'justification_by_omni_judge',   
})
#df_kimi_merged_tagged = df_kimi_merged_tagged.drop(columns=['correctness'])
df_kimi_merged_tagged = df_kimi_merged_tagged[[
    'id',
    'difficulty',
    'domain',   
    'problem',
    'reference_answer',
    'final_answer_by_omni_judge',
    'justification_by_omni_judge',
    'score_omni_judge',
    'final_answer_by_gpt5mini',
    'justification_by_gpt5mini',
    'score_gpt5mini'
]]
df_kimi_merged_tagged = df_kimi_merged_tagged[df_kimi_merged_tagged['score_omni_judge'] != df_kimi_merged_tagged['score_gpt5mini']]
#df_kimi_merged_tagged.to_excel('kimi_tagged_disagreements_to_annotate.xlsx', index=False, engine='openpyxl')

# Gemini
df_gemini_merged_tagged = df_gemini_merged_tagged.rename(columns={
    'judged_correct': 'score_gpt5mini',
    'score': 'score_omni_judge',
    'extracted_final_answer': 'final_answer_by_gpt5mini',
    'final_answer': 'final_answer_by_omni_judge',
    'answer': 'reference_answer',
    'Equivalence Judgement': 'justification_by_gpt5mini',
    'Justification': 'justification_by_omni_judge',   
})
#df_gemini_merged_tagged = df_gemini_merged_tagged.drop(columns=['correctness'])
df_gemini_merged_tagged = df_gemini_merged_tagged[[
    'id',
    'difficulty',
    'domain',   
    'problem',
    'reference_answer',
    'final_answer_by_omni_judge',
    'justification_by_omni_judge',
    'score_omni_judge',
    'final_answer_by_gpt5mini',
    'justification_by_gpt5mini',
    'score_gpt5mini'
]]
df_gemini_merged_tagged = df_gemini_merged_tagged[df_gemini_merged_tagged['score_omni_judge'] != df_gemini_merged_tagged['score_gpt5mini']]
#df_gemini_merged_tagged.to_excel('gemini_tagged_disagreements_to_annotate.xlsx', index=False, engine='openpyxl')


# --------------------------------------- Analyze the annotated excel files --------------------------------------- #



#Omni-Math-2-Tagged disagreements
df_tagged_claude = pd.read_excel('claude_tagged_disagreements_to_annotate.xlsx')
df_tagged_deepseek = pd.read_excel('deepseek_tagged_disagreements_to_annotate.xlsx')
df_tagged_kimi = pd.read_excel('kimi_tagged_disagreements_to_annotate.xlsx')
df_tagged_gemini = pd.read_excel('gemini_tagged_disagreements_to_annotate.xlsx')
df_tagged_gpt5 = pd.read_excel('gpt5_tagged_disagreements_to_annotate.xlsx')
total_disagreements_tagged = len(df_tagged_claude) + len(df_tagged_deepseek) + len(df_tagged_kimi) + len(df_tagged_gemini) + len(df_tagged_gpt5)
print(f"Total disagreements on tagged data: {total_disagreements_tagged}")

print(df_tagged_claude.head())
oj_wrong_claude = df_tagged_claude[df_tagged_claude['Judge that is wrong'] == 'Omni-Judge']
g5_wrong_claude = df_tagged_claude[df_tagged_claude['Judge that is wrong'] == 'gpt5mini']
oj_wrong_deepseek = df_tagged_deepseek[df_tagged_deepseek['Judge that is wrong']=='Omni-Judge']
g5_wrong_deepseek = df_tagged_deepseek[df_tagged_deepseek['Judge that is wrong']=='gpt5mini']
oj_wrong_kimi = df_tagged_kimi[df_tagged_kimi['Judge that is wrong']=='Omni-Judge']
g5_wrong_kimi = df_tagged_kimi[df_tagged_kimi['Judge that is wrong']=='gpt5mini']
oj_wrong_gemini = df_tagged_gemini[df_tagged_gemini['Judge that is wrong']=='Omni-Judge']
g5_wrong_gemini = df_tagged_gemini[df_tagged_gemini['Judge that is wrong']=='gpt5mini']
oj_wrong_gpt5 = df_tagged_gpt5[df_tagged_gpt5['Judge that is wrong']=='Omni-Judge']
g5_wrong_gpt5 = df_tagged_gpt5[df_tagged_gpt5['Judge that is wrong']=='gpt5mini']
total_oj_wrong = len(oj_wrong_claude) + len(oj_wrong_deepseek) + len(oj_wrong_kimi) + len(oj_wrong_gemini) + len(oj_wrong_gpt5)
total_g5_wrong = len(g5_wrong_claude) + len(g5_wrong_deepseek) + len(g5_wrong_kimi) + len(g5_wrong_gemini) + len(g5_wrong_gpt5)
print(f"Total Omni-Judge wrong: {total_oj_wrong}, Total gpt5mini wrong: {total_g5_wrong}")

#Omni-Math-2-Filtered disagreements
df_filtered_gpt5 = pd.read_excel('gpt5_filtered_disagreements_to_annotate.xlsx')
oj_wrong_gpt5_filtered = df_filtered_gpt5[df_filtered_gpt5['Wrong judge']=='Omni-Judge']
g5_wrong_gpt5_filtered = df_filtered_gpt5[df_filtered_gpt5['Wrong judge']=='gpt5mini']
total_oj_wrong_filtered = len(oj_wrong_gpt5_filtered)
total_g5_wrong_filtered = len(g5_wrong_gpt5_filtered)
print(f"Filtered data - Omni-Judge wrong: {total_oj_wrong_filtered}, gpt5mini wrong: {total_g5_wrong_filtered}, total disagreements: {len(df_filtered_gpt5)}")

oj_failed_equivalence = oj_wrong_gpt5_filtered[oj_wrong_gpt5_filtered['Failed to assess equivalence']==1]
g5_failed_equivalence = g5_wrong_gpt5_filtered[g5_wrong_gpt5_filtered['Failed to assess equivalence']==1]
oj_wrong_extraction = oj_wrong_gpt5_filtered[oj_wrong_gpt5_filtered['Wrong extraction']==1]
g5_wrong_extraction = g5_wrong_gpt5_filtered[g5_wrong_gpt5_filtered['Wrong extraction']==1]
oj_no_instructions = oj_wrong_gpt5_filtered[oj_wrong_gpt5_filtered['Didn\'t follow instructions']==1]
g5_no_instructions = g5_wrong_gpt5_filtered[g5_wrong_gpt5_filtered['Didn\'t follow instructions']==1]
oj_too_obedient = oj_wrong_gpt5_filtered[oj_wrong_gpt5_filtered['Too obedient']==1]
g5_too_obedient = g5_wrong_gpt5_filtered[g5_wrong_gpt5_filtered['Too obedient']==1]
oj_not_clear = oj_wrong_gpt5_filtered[oj_wrong_gpt5_filtered['Not clear']==1]
g5_not_clear = g5_wrong_gpt5_filtered[g5_wrong_gpt5_filtered['Not clear']==1]
oj_wrong_reference = oj_wrong_gpt5_filtered[oj_wrong_gpt5_filtered['Reference answer error']==1]
g5_wrong_reference = g5_wrong_gpt5_filtered[g5_wrong_gpt5_filtered['Reference answer error']==1]
oj_wrong_problem = oj_wrong_gpt5_filtered[oj_wrong_gpt5_filtered['Problem statement error']==1]
g5_wrong_problem = g5_wrong_gpt5_filtered[g5_wrong_gpt5_filtered['Problem statement error']==1]
print(f"Omni-Judge failed equivalence: {len(oj_failed_equivalence)}, gpt5mini failed equivalence: {len(g5_failed_equivalence)}")
print(f"Omni-Judge wrong extraction: {len(oj_wrong_extraction)}, gpt5mini wrong extraction: {len(g5_wrong_extraction)}")
print(f"Omni-Judge no instructions: {len(oj_no_instructions)}, gpt5mini no instructions: {len(g5_no_instructions)}")
print(f"Omni-Judge too obedient: {len(oj_too_obedient)}, gpt5mini too obedient: {len(g5_too_obedient)}")
print(f"Omni-Judge not clear: {len(oj_not_clear)}, gpt5mini not clear: {len(g5_not_clear)}")
print(f"Omni-Judge wrong reference answer: {len(oj_wrong_reference)}, gpt5mini wrong reference answer: {len(g5_wrong_reference)}")
print(f"Omni-Judge wrong problem statement: {len(oj_wrong_problem)}, gpt5mini wrong problem statement: {len(g5_wrong_problem)}")
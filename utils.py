import json
import pandas as pd
import numpy as np

def parse_report(report):
    parts = report.split("## ")
    data = {}
    
    for part in parts[1:]:  
        lines = part.strip().split("\n")
        title = lines[0].strip() 
        content = "\n".join(lines[1:]).strip()  
        
        if title == "Justification":
            data[title] = content
        else:
            data[title] = lines[1].strip() if len(lines) > 1 else ''
    
    return data

def get_dataframe_gpt5mini(file_path):
        df = pd.read_json(file_path, lines=True)
        # extract judgement
        records = []
        for i, row in df.iterrows():
            response = row['response']['body']['output'][1]['content'][0]['text']
            extracted_final_answer = json.loads(response)['extracted_final_answer']
            judge = json.loads(response)['reasoning']
            judgement = json.loads(response)['correct']
            if judgement == 'yes':
                judged_correct = 1
                correctness = True
            else:
                judged_correct = 0
                correctness = False
            record = {
                'id': row['custom_id'],
                'gpt5mini': response,
                'extracted_final_answer': extracted_final_answer,
                'Equivalence Judgement': judge,
                'correctness': correctness,
                'judged_correct': judged_correct,
            }
            records.append(record)
        return pd.DataFrame(records)


def get_dataframe_reasoning_models(file, reference_file):
    records = []
    count = 0
    with open(reference_file, "r") as ref:
        lines = ref.readlines()
        with open(file, 'r') as file:
            for idx, line in enumerate(file):
                count += 1
                json_obj = json.loads(line)
                ref_object = json.loads(lines[idx])
                info = parse_report(json_obj['omni-judge'])
                if info == {}:
                    continue
                try:
                    correctness = info['Equivalence Judgement']
                    if correctness == 'TRUE':
                        records.append({'id': ref_object['id'], 'difficulty': json_obj['difficulty'], 'domain': json_obj['domain'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'final_answer': info['Student Final Answer'], 'score': 1, 'correctness': True, 'Justification': info['Justification']})
                    else:
                        records.append({'id': ref_object['id'], 'difficulty': json_obj['difficulty'], 'domain': json_obj['domain'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'final_answer': info['Student Final Answer'], 'score': 0, 'correctness': False, 'Justification': info['Justification']})
                except:
                    continue
        
    Data_df = pd.DataFrame(records)
    return Data_df

def parse_domain(domain_tree):
    if not isinstance(domain_tree, str):
        return domain_tree  
    parts = domain_tree.split("->")
    if len(parts) < 2:
        return np.nan  
    return parts[1].strip()

def domain_performance(file, reference_file, judge):
    if judge ==  'Omni-Judge':
        Data_df = get_dataframe_reasoning_models(file, reference_file)
    
    elif judge == 'gpt5mini':
        Data_df = get_dataframe_gpt5mini(file)

        #Join with reference file to get all meta data
        ref_df = pd.read_json(reference_file, lines=True)
        Data_df = Data_df.merge(ref_df[['id', 'domain']], left_on='id', right_on='id', how='left')
    else:
        raise ValueError("Judge must be either 'Omni-Judge' or 'gpt5mini'")

    #The multi-domain questions are taken into account for each domain by using explode
    Data_df = Data_df.explode('domain').reset_index()
    Data_df['domain'] = Data_df['domain'].apply(parse_domain)

    #Deduplicate the data as some multi-domain questions might have the same primary domain
    Data_df_deduplicated = Data_df.drop_duplicates(subset=None)
    Data_df_deduplicated = Data_df_deduplicated.dropna() #There is one empty domain tree that gets value nan by parse_domain

    #Join the calculus and precalculus domains
    Data_df_deduplicated['domain'] = Data_df_deduplicated['domain'].apply(lambda x: 'Calculus' if x == 'Precalculus' else x)

    Domains = Data_df_deduplicated['domain'].unique()
    domain_performance = {domain: {'correct': 0, 'total': 0, 'accuracy': 0} for domain in Domains}
    for domain in Domains:
        domain_df = Data_df_deduplicated[Data_df_deduplicated['domain'] == domain]
        domain_performance[domain]['total'] = len(domain_df)
        domain_performance[domain]['correct'] = len(domain_df[domain_df['correctness'] == True])
        domain_performance[domain]['accuracy'] = (domain_performance[domain]['correct'] / domain_performance[domain]['total']) * 100
 
    return domain_performance

def difficulty_performance(file, reference_file, judge):
    if judge ==  'Omni-Judge':
        Data_df = get_dataframe_reasoning_models(file, reference_file)

    elif judge == 'gpt5mini':
        Data_df = get_dataframe_gpt5mini(file)

        #Join with reference file to get all meta data
        ref_df = pd.read_json(reference_file, lines=True)
        Data_df = Data_df.merge(ref_df[['id', 'difficulty']], left_on='id', right_on='id', how='left')
    else:
        raise ValueError("Judge must be either 'Omni-Judge' or 'gpt5mini'")

    #Perform q-cut to divide the data into equally sized difficulty tiers
    Data_df['difficulty'] = pd.qcut(Data_df['difficulty'], 4, labels=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

    difficulty_levels = Data_df['difficulty'].unique()
    difficulty_performance = {difficulty: {'correct': 0, 'total': 0, 'accuracy': 0} for difficulty in difficulty_levels}
    for difficulty in difficulty_levels:
        difficulty_df = Data_df[Data_df['difficulty'] == difficulty]
        difficulty_performance[difficulty]['total'] = len(difficulty_df)
        difficulty_performance[difficulty]['correct'] = len(difficulty_df[difficulty_df['correctness'] == True])
        difficulty_performance[difficulty]['accuracy'] = (difficulty_performance[difficulty]['correct'] / difficulty_performance[difficulty]['total']) * 100
    
    return difficulty_performance

def get_dataframe_reasoning_models_with_tags(file, reference_file):
    records = []
    count = 0
    with open(reference_file, "r") as ref:
        lines = ref.readlines()
        with open(file, 'r') as file:
            for idx, line in enumerate(file):
                count += 1
                json_obj = json.loads(line)
                ref_object = json.loads(lines[idx])
                info = parse_report(json_obj['omni-judge'])
                if info == {}:
                    continue
                try:
                    correctness = info['Equivalence Judgement']
                    if correctness == 'TRUE':
                        records.append({'id': ref_object['id'], 'difficulty': json_obj['difficulty'], 'domain': json_obj['domain'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'final_answer': info['Student Final Answer'], 'score': 1, 'correctness': True, 'Justification': info['Justification'], 'tags': ref_object['tags']})
                    else:
                        records.append({'id': ref_object['id'], 'difficulty': json_obj['difficulty'], 'domain': json_obj['domain'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'final_answer': info['Student Final Answer'], 'score': 0, 'correctness': False, 'Justification': info['Justification'], 'tags': ref_object['tags']})
                except:
                    continue
        
    Data_df = pd.DataFrame(records)
    return Data_df
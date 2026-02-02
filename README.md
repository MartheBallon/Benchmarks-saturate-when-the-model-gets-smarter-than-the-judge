# Benchmarks saturate when the model gets smarter than the judge

This repository contains the code to 'Benchmarks saturate when the model becomes smarter than the judge' by Marthe Ballon, Andres Algaba, Brecht Verbeken and Vincent Ginis ([arXiv link](https://arxiv.org/abs/2601.19532v1)).

Benchmarks are important tools for tracking progress in the development of large language models (LLMs). However, inaccuracies in datasets and evaluation methods often undermine their effectiveness. Here, we present Omni-MATH-2: a manually revised version of the original Omni-MATH dataset that preserves its size (n = 4,428) while significantly improving LaTeX compilability, solvability, and verifiability. A total of 647 problems were edited (14.6%) and 247 were tagged as non-standard (5.6%). This means they contain images, request estimations or proofs, or are degenerate (e.g. duplicate, no reference answer, empty problem etc). We have released multiple evaluation-ready subsets, notably Omni-MATH-2-Filtered (n = 4,181), from which the tagged non-standard questions have been excluded to ensure suitability for judging exact answers. 

## Overview of the cleaning process
![Cleaning process](Figures/Cleaning_process.pdf)

## Overview of the data
The Omni-MATH-2 dataset is available at (https://huggingface.co/datasets/martheballon/Omni-MATH-2). All other data necessary to replicate the results are available at (10.5281/zenodo.18380308).
   
```
.
├── claude/
│   ├── claude_filtered_judged_by_gpt5mini.jsonl       # GPT-5 mini as a judge outputs on Omni-MATH-2-Filtered 
│   ├── claude_filtered_judged_by_oj.jsonl             # Omni-Judge outputs on Omni-MATH-2-Filtered 
│   ├── claude_tagged_judged_by_gpt5mini.jsonl         # GPT-5 mini as a judge outputs on Omni-MATH-2-Tagged
│   └── claude_tagged_judged_by_oj.jsonl               # Omni-Judge outputs on Omni-MATH-2-Tagged         
├── deepseek/
│   ├── deepseek_filtered_judged_by_gpt5mini.jsonl
│   ├── deepseek_filtered_judged_by_oj.jsonl
│   ├── deepseek_tagged_judged_by_gpt5mini.jsonl
│   └── deepseek_tagged_judged_by_oj.jsonl 
├── gemini/
│   ├── ...                                          
├── gpt-5/
│   ├── ...
├── kimi/
│   ├── ...
├── Analysis/                                          
│   ├── figures_main.ipynb                           # Notebook to generate figures
│   ├── compute_granular_performance.py              # Compute domain and difficulty specific accuracy (based on the code in Gao et al. 2024)
│   ├── Table1.py                                    # Script to generate Table 1 
│   ├── TableA1.py
│   ├── TableA2.py
│   └── utils.py              
├── Figures/
│   ├── Cleaning_process.pdf                        # Overview of the cleaning process
│   ├── Figure 3.pdf       
│   ├── Figure 5.pdf             
│   ├── Example_estimation.pdf                      # Example of the evaluation pipeline (dataset, model, judge) on a problem with tag 'estimation'   
│   ├── Example_proof.pdf
│   ├── Example_missing_image.pdf
│   ├── Example_should_delete.pdf
│   ├── Example_easy_equivalence.pdf                # Example of a judge disagreemend where equivalence with the reference answer is easy to assess
│   ├── Example_hard_equivalence.pdf
│   └── Example_wrong_reference_answer.pdf            
└── requirements.txt
```

## Datasets

Omni-MATH-2, a manually revised version of Omni-MATH, comprising a clean, exact answer subset Omni-MATH-2-Filtered and a tagged, non-standard subset Omni-MATH-2-Tagged (questions requiring proofs, estimation, images). To obtain the two subsets execute the following commands.
```bash
# Create subsets Omni-MATH-2-Filtered and Omni-MATH-2-Tagged
omni_math_2 = pd.read_json("Omni-Math-2.jsonl", lines=True)

omni_math_2_tagged = omni_math_2[omni_math_2['tags'].apply(lambda x: len(x) > 0)]
#omni_math_2_tagged.to_json("Omni-Math-2-Tagged.jsonl", lines=True, orient='records')

omni_math_2_filtered = omni_math_2[omni_math_2['tags'].apply(lambda x: len(x) == 0)]
#omni_math_2_filtered.to_json("Omni-Math-2-Filtered.jsonl", lines=True, orient='records')
```

## Models
- Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`, 64,000 output tokens, 25,000 thinking budget, Claude Batch API)
- DeepSeek V3.2 (`deepseek-reasoner`, 64,000 output tokens (shared), implicit thinking budget, DeepSeek API)
- Gemini 3 Pro (`gemini-3-pro-preview`, 64,000 output tokens, high thinking level, no thinking budget, Gemini Batch API)
- GPT-5 (`gpt-5-2025-08-07`, medium reasoning effort, no token limit, OpenAI Batch API)
- Kimi K2 Thinking (`kimi-k2-thinking`, 128,000 output tokens via `max_tokens`, 256,000 context window, Moonshot API)

## Example of how dataset errors can propagate through the evaluation pipeline
![Example_image](Figures/Example_missing_image.pdf)


## Citation
If you find our dataset helpful, you are welcome to cite our paper.
```
@misc{ballon2026benchmarkssaturatewhen,
      title={Benchmarks Saturate When The Model Becomes Smarter Than The Judge}, 
      author={Ballon, Marthe and Algaba, Andres and Verbeken, Brecht and Ginis, Vincent},
      year={2026},
      eprint={2601.19532},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.19532}, 
}
```

## Acknowledgements
We thank the authors and contributors of the original Omni-MATH dataset. Their release of the benchmark and verifier Omni-Judge enabled the analyses in this paper.
```
@misc{gao2024omnimathuniversalolympiadlevel,
      title={Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models}, 
      author={Bofei Gao and Feifan Song and Zhe Yang and Zefan Cai and Yibo Miao and Qingxiu Dong and Lei Li and Chenghao Ma and Liang Chen and Runxin Xu and Zhengyang Tang and Benyou Wang and Daoguang Zan and Shanghaoran Quan and Ge Zhang and Lei Sha and Yichang Zhang and Xuancheng Ren and Tianyu Liu and Baobao Chang},
      year={2024},
      eprint={2410.07985},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.07985}, 
}
```

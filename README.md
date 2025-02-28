# Nuance Matters: Probing Epistemic Consistency in Causal Reasoning 
This is the repository containing data, scripts and results, for the paper entitled _''Nuance Matters: Probing Epistemic Consistency in Causal Reasoning''_, by Shaobo Cui, Junyou Li, Luca Mouchel, Yiyang Feng and Boi Faltings. This project aims to investigate whether LLMs can **generate** and then later **discern** defeaters or supporters with different intensity. This project will be published at the 39th Annual AAAI Conference on Artificial Intelligence.
 
There are mainly three steps for the experiments: 

## 1. Generation 
The generation phase involves closed-source models, including GPT series, Claude, and Gemini, and open-source models such as LLaMA, Gemma, Phi, etc. 
### 1.1. Generation with Closed-Source Models
The default option generates intermediates in pairwise fashion, given a cause-effect pair and either a defeater or a supporter. This might be very helpful for open-source models like LLaMA, Gemma, Phi, which are less satisfyingly instructed. 
```bash
python closedSourceScripts/generation_prompt_{series_name}_pair.py --model_name {model_name}
```
Please note there is another option for generating these intermediates, which is to generate all ten intermediates as once while only given the cause-effect pair.
```bash
python closedSourceScripts/generation_prompt_{series_name}.py --model_name {model_name}
```
### 1.2. Generation with Open-Source Models
We prompt open-source models with significant smaller size compared to the closed-source models. We prompt in a zero-shot setting, meaning models are not trained to output a particular format. As such, we have different scripts for each model to tailor to each model's outputs. We prompt the LLaMA 2 and 3, Gemma and Phi-3. 

To generate using gemma, run the following command:
```bash
python openSourceGenerations/gemma/generate.py --model-id=google/gemma-1.1-<2 or 7>b-it
```
For LLaMA 2:
```bash
python openSourceGenerations/llama/generate.py --model-id=meta-llama/Llama-2-<7, 13 or 70>b-chat-hf
```

LLaMA 3: 
```bash
python openSourceGenerations/llama3/generate.py --model-id=meta-llama/Meta-Llama-3-<8 or 70>B-Instruct
```
Phi-3:
```bash
python openSourceGenerations/phi/generate_phi3-<3.8, 7 or 14>b.py 
```



## 2. Ranking

Ranking consists of two different ways: prompt and by conditional probability. 

### 2.1. Prompting

#### Closed-Source models
Run the following commands for ranking outputs based on each series of the closed-source models. 
```bash
python closedSourceScripts/ranking_prompt_{series_name}.py --model_name {model_name}
```
#### Open-Source Models
For each family of open-source models, you can run:
```bash
python openSourceGenerations/<gemma, llama, llama3, phi>/ranking_<gemma, llama, llama3, phi>-<model size>b.py
```



### 2.2. Ranking Based on Conditional Probability

<span style="color: red">@Shaobo Here is Shaobo might need to develop. Thanks in advance!</span>.




## 3. Evaluation

### Important Output Format in CSV Format
In step **2.**, we ranked the intermediates we generated in step **1.**, based on the intensities. The outputs of step **2.** are given by: 
```bash
g-SD2,g-SD1,g-OD,g-WD1,g-WD2,g-WS2,g-WS1,g-OS,g-SS1,g-SS2
0,1,3,4,2,5,6,8,7,9
```
| Header | Meaning                                    |
|--------|--------------------------------------------|
| g-SD2  | The strongest defeater generated by model. |
| g-SD1  | The stronger defeater generated by model.  |
| g-OD   | The ordinary defeater generated by model.  |
| g-WD1  | The weaker defeater generated by model.    |
| g-WD2  | The weakest defeater generated by model.   |
| g-WS2  | The weakest supporter generated by model.  |
| g-WS1  | The weaker supporter generated by model.   |
| g-OS   | The ordinary supporter generated by model. |
| g-OS1  | The stronger supporter generated by model. |
| g-OS2  | The strongest supporter generated by model.|

Ideally, the perfect model would rank the intermediates it generated itself in the right order. 
In practice, that is not the case.

Run the following command to get the evaluation results: 
```bash
python evaluation_main.py --input-file=<path to ranking csv>
```


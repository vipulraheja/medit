# mEdIT: Multilingual Text Editing via Instruction Tuning  <br />(Accepted at NAACL 2024)

This repository provides datasets, models and code for mEdIT instruction-tuned text editing model(s), with the official implementation of the following paper:
> [mEdIT: Multilingual Text Editing via Instruction Tuning](https://arxiv.org/abs/2402.16472v1) <br>
> [Vipul Raheja](https://github.com/vipulraheja), [Dimitris Alikaniotis](https://github.com/dimalik), [Vivek Kulkarni](https://github.com/viveksck), [Bashar Alhafni](https://github.com/balhafni) and [Dhruv Kumar](https://github.com/ddhruvkr)

Our code is based on Hugging Face `transformers`.

## Installation
1. Clone the repository
   ```
   git clone https://github.com/vipulraheja/medit.git
   ```
   
2. Run the setup script
   ```
   $ cd medit
   $ sh setup_env.sh
   ```

## Data
Available on [Hugging Face](https://huggingface.co/datasets/grammarly/medit).
Example data point:

```
{
  "instance":867453,
  "task":"gec",
  "language":"english",
  "lang":"en",
  "dataset":"lang8.bea19",
  "src":"Fix grammar in this sentence: Luckily there was no damage for the earthquake .",
  "full_prompt":"### 命令:\nこの文の文法上の誤りを修正してください\n### 入力:\nLuckily there was no damage for the earthquake .\n### 出力:\n\nLuckily there was no damage from the earthquake two years ago .",
  "prompt":"### 命令:\nこの文の文法上の誤りを修正してください\n### 入力:\nLuckily there was no damage for the earthquake .\n### 出力:\n\n",
  "text":"Luckily there was no damage from the earthquake two years ago ."
}
```

Please note that this dataset contains XX instances (as opposed to the XX instances we used in the paper). This is because this public release includes only the instances that were acquired and curated from publicly available datasets. Specifically, it is missing roughly XX instances in training and XX instances in validation data from Simplification and Formality Transfer tasks due to licensing restrictions.

## Code
### Training
Script for the `mEdIT-xxl` model. 
```
sh train/train_medit_xxl.sh
```
All other models models can be trained by making the corresponding changes to this script. 

## Model

#### Model checkpoint
Due to the quality, we only publicly release the best-performing model checkpoint to [Hugging Face](https://huggingface.co/grammarly). 

| Model         | Params        | 
| :-------------|:-------------  |
| mEdIT-large (TBA)    | 1.2B  | 
| mEdIT-xl (TBA)    | 7B  | 
| [mEdIT-xxl] (https://huggingface.co/grammarly/medit-xxl)    | 13B  | 


#### Example Usage:
You can directly load our models using [Hugging Face Transformers](https://github.com/huggingface/transformers).
```python

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("grammarly/medit-xxl")
model = T5ForConditionalGeneration.from_pretrained("grammarly/medit-xxl")

input_text = '### 命令:\n文章を文法的にする\n### 入力:\nI has small cat ,\n### 出力:\n\n'
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=20)
edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(edited_text)
# --> I have a small cat ,
```

## Citation
```
@article{raheja2023medit,
      title={mEdIT: Multilingual Text Editing via Instruction Tuning}, 
      author={Vipul Raheja and Dimitris Alikaniotis and Vivek Kulkarni and Bashar Alhafni and Dhruv Kumar},
      url={https://arxiv.org/abs/2402.16472v1},
      year={2024},
      eprint={2402.16472},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

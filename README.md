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

### Example data point:

```
{
  "instance":999999,
  "task":"gec",
  "language":"english",
  "lang":"en",
  "dataset":"lang8.bea19",
  "src":"Luckily there was no damage for the earthquake .",
  "refs": ['Luckily there was no damage from the earthquake .'],
  "tgt":"Luckily there was no damage from the earthquake .",
  "prompt":"この文の文法上の誤りを修正してください: Luckily there was no damage for the earthquake .",
}
```

Note that for the mEdIT models, the `prompt` was formatted as follows: 
(e.g. for a Japanese-prompted editing for English text)
```
### 命令:\nこの文の文法上の誤りを修正してください\n### 入力:\nLuckily there was no damage for the earthquake .\n### 出力:\n\n
```
Details about the added keywords ("Instruction", "Input", "Output") can be found in the Appendix or on the mEdIT model cards. 


### Data Fields
* `instance`: instance ID
* `language`: Language of input and edited text 
* `lang`: Language code in ISO-639-1
* `dataset`: Source of the current example
* `task`: Text editing task for this instance
* `src`: input text (formatted as `instruction: input_text`)
* `prompt`: Full prompt (instruction + input) for training the models
* `text`: output text


Please note that this dataset contains 102k instances (as opposed to the 190k instances we used in the paper to train our models). This is because this public release includes only the instances that were acquired and curated from publicly available datasets. More details on dataset sources can be found in the paper or on the [Hugging Face Dataset card](https://huggingface.co/datasets/grammarly/medit). 

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
| [mEdIT-xl](https://huggingface.co/grammarly/medit-xl)    | 7B  | 
| [mEdIT-xxl](https://huggingface.co/grammarly/medit-xxl)    | 13B  | 


#### Example Usage:
You can directly load our models using [Hugging Face Transformers](https://github.com/huggingface/transformers).
```python

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("grammarly/medit-xxl")
model = AutoModelForCausalLM.from_pretrained("grammarly/medit-xxl")

# English GEC using Japanese instructions
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

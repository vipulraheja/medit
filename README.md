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
* `src`: input text 
* `refs`: reference texts
* `tgt`: output text
* `prompt`: Full prompt (instruction + input) for training the models

Please note that this dataset contains 102k instances (as opposed to the 190k instances we used in the paper to train our models). This is because this public release includes only the instances that were acquired and curated from publicly available datasets. More details on dataset sources can be found in the paper or on the [Hugging Face Dataset card](https://huggingface.co/datasets/grammarly/medit). 

## Code
### Training
We provide a minimal script for training the `mEdIT-xl` model. We use the standard [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) from Hugging Face.
```
sh train/train_medit_minimal.sh
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
@inproceedings{raheja-etal-2024-medit,
    title = "m{E}d{IT}: Multilingual Text Editing via Instruction Tuning",
    author = "Raheja, Vipul  and
      Alikaniotis, Dimitris  and
      Kulkarni, Vivek  and
      Alhafni, Bashar  and
      Kumar, Dhruv",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.56",
    doi = "10.18653/v1/2024.naacl-long.56",
    pages = "979--1001",
    abstract = "We introduce mEdIT, a multi-lingual extension to CoEdIT {--} the recent state-of-the-art text editing models for writing assistance. mEdIT models are trained by fine-tuning multi-lingual large, pre-trained language models (LLMs) via instruction tuning. They are designed to take instructions from the user specifying the attributes of the desired text in the form of natural language instructions, such as {``}Grammatik korrigieren{''} (German) or {``}이 텍스 트를 단순화{''} (Korean). We build mEdIT by curating data from multiple publicly available human-annotated text editing datasets for three text editing tasks (Grammatical Error Correction (GEC), Text Simplification, and Paraphrasing) across diverse languages belonging to six different language families. We detail the design and training of mEdIT models and demonstrate their strong performance on many multi-lingual text editing benchmarks against other multilingual LLMs. We also find that mEdIT generalizes effectively to new languages over multilingual baselines. We publicly release our data, code, and trained models.",
}
```

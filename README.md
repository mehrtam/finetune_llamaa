#  TinyLLaMA Fine-Tuning with LoRA (PEFT)

This repository demonstrates how to fine-tune the [TinyLLaMA-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model using **Low-Rank Adaptation (LoRA)** via the Hugging Face [PEFT](https://github.com/huggingface/peft) library. It uses the **WikiText-2** dataset for training and evaluates the model's ability to predict the next word in both automatic and custom prompts.

---

##  Features

-  Fine-tunes a 1.1B parameter TinyLLaMA model efficiently using LoRA
-  Uses WikiText-2 for training and testing
-  Includes both automatic and custom evaluation with top-k accuracy
-  Saves and compresses the trained model for easy deployment
-  LoRA applied only to attention projection layers for efficiency

---

Installation

```bash
pip install -q transformers accelerate peft datasets bitsandbytes sentencepiece
Authenticate with Hugging Face if needed:

python
Copy
Edit
from huggingface_hub import notebook_login
notebook_login()
 Model & Tokenizer
We use:

python
Copy
Edit
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
LoRA configuration:

python
Copy
Edit
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)
 Dataset
Dataset: WikiText-2 Raw

Preprocessing: Removed short samples (len < 30), tokenized with max_length=128

Labels: input_ids are duplicated for next-token prediction (causal_lm)

DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
Evaluation
Automatic Evaluation (Top-k Accuracy)
10 runs × 30 samples from WikiText test set

Top-1, Top-3, and Top-5 accuracy computed on next-token prediction

Custom Prompt Evaluation
30 handcrafted prompts with known next-token

Checks if the expected token appears in top-1, top-3, or top-5 predictions

## Example:

Prompt: "The capital of France is"
Expected: "Paris"
Top-5 Predictions: ["Paris", "Berlin", "London", "Rome", "Madrid"]
Saving & Export
model.save_pretrained("llama_peft_lora")
tokenizer.save_pretrained("llama_peft_lora")
!zip -r llama_peft_lora.zip llama_peft_lora
Download from Colab:
from google.colab import files
files.download("llama_peft_lora.zip")
Push to Hugging Face Hub (Optional)

Copy
Edit
from huggingface_hub import login
login()

model.push_to_hub("your-username/llama_peft_lora")
tokenizer.push_to_hub("your-username/llama_peft_lora")

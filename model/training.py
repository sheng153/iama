import sqlite3
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

IGNORE_LETTERS = ['?','!','.',',']

conn = sqlite3.connect("/data/dev.db")
rows = conn.execute("SELECT role, content FROM conversation ORDER BY timestamp").fetchall()
conn.close()

examples = [{"text": f"{r[0]} : {r[1]}\n"} for r in rows]
ds = Dataset.from_list(examples)

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def tokenize(ex):
    return tokenizer(ex["text"], truncation=True, max_length=512)

ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="/data/finetuned-phi2",
)

training = Trainer(
    model = model,
    args = training_args,
    train_dataset = ds_tok,
    tokenizer = tokenizer
)

trainer.train()
trainer.save_model("/data/phi2-chatbot")
tokenizer.save_pretrained("/data/phi2-chatbot")

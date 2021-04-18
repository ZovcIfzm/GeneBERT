from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

DATA_DIR = "./word_tokenized"
MODEL_DIR = "DeepInflam"

paths = [str(x) for x in Path(DATA_DIR).glob("**/*.tsv")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=1000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(MODEL_DIR)

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    MODEL_DIR+"/vocab.json",
    MODEL_DIR+"/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print("three four encoded: ", tokenizer.encode("three four"))

# 3. Train a language model from scratch
import torch

# define a config
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=1000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR, max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

print("model num param:", model.num_parameters())

# build training dataset
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=DATA_DIR+"/trainValidCorpus.tsv",
    block_size=128,
)

# define data_colator (batch different samples of the dataset together into object PyTorch knows how to perform backprop on)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# initialize trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(MODEL_DIR)

# Check that the LM actually trained
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR
)

print(fill_mask("three four <mask>."))
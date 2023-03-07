# -*- coding: utf-8 -*-
import os
import glob
os.environ["WANDB_PROJECT"] = "ul2_pretrain"
from transformers import (AutoTokenizer, Seq2SeqTrainer, MT5ForConditionalGeneration, MT5Config,
                          Seq2SeqTrainingArguments, trainer, AutoModelForSeq2SeqLM)
from collate_fn import DataCollatorForUL2
from tests.test_dataset import ZstDataset
from tests import utils

if __name__ == "__main__":
    model_name = "theblackcat102/mt0-chat-large"
    model_saved_name = model_name.split("/")[-1]
    import wandb
    wandb.init(
        project="ul2_pretrain",
        name=model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'bos_token': '<s>'})

    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # model.resize_token_embeddings(len(tokenizer))

    config = MT5Config.from_pretrained(model_name)
    config.vocab_size = len(tokenizer)
    config.dropout_rate = 0.1
    model = MT5ForConditionalGeneration(config)

    print(model.config)
    print(tokenizer.convert_tokens_to_ids)
    collate_fn = DataCollatorForUL2(tokenizer)
    train_dataset = ZstDataset(list(glob.glob('/mnt/ssd/pythia/*.jsonl.zst')), tokenizer, max_length=600)
    val_dataset = ZstDataset('/mnt/ssd/pythia_val/val.jsonl.zst', tokenizer, max_length=600)

    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-ul2",
        fp16=True,
        # deepspeed="zero2_config.json",
        max_steps=100000,
        warmup_steps=4000,
        learning_rate=1e-5,
        label_smoothing_factor=0,
        optim="adamw_hf",
        gradient_checkpointing=True,
        dataloader_num_workers=9,
        gradient_accumulation_steps=50,
        per_device_train_batch_size=7,
        per_device_eval_batch_size=5,
        weight_decay=0.01,
        max_grad_norm=2,
        logging_steps=10,
        save_total_limit=4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    trainer.train()

import torch
from transformers import Trainer, set_seed
from table_dataloader import Data2TextTrainDataset, DataCollatorForData2TextTrain
import os
from arguments import *
import argparse
from arguments import MODEL_CLASSES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="webnlg")
    
    parser.add_argument("--train_src", type=str, default="afs/webnlg/data/rewrite0/train/src.txt")
    parser.add_argument("--train_gold", type=str, default="afs/webnlg/data/rewrite0/train/gold.txt")
    parser.add_argument("--train_table", type=str, default="afs/webnlg/data/rewrite0/train/table.json")

    parser.add_argument("--eval_src", type=str, default="afs/webnlg/data/rewrite0/test/src.txt")
    parser.add_argument("--eval_gold", type=str, default="afs/webnlg/data/rewrite0/test/gold.txt")
    parser.add_argument("--eval_table", type=str, default="afs/webnlg/data/rewrite0/test/table.json")

    parser.add_argument("--cand_train_path", type=str, default="afs/webnlg/result/bart_tableno_rewrite0/train/sample.txt")
    parser.add_argument("--cand_train_metrics_path", type=str, default="afs/webnlg/metrics/bart_tableno_rewrite0/train/sample.json")

    parser.add_argument("--cand_eval_path", type=str, default="afs/webnlg/result/bart_tableno_rewrite0/test/out.txt")
    parser.add_argument("--cand_eval_metrics_path", type=str, default="afs/webnlg/metrics/bart_tableno_rewrite0/test/every.json")
    
    parser.add_argument("--model_type", type=str, default="bart")
    parser.add_argument("--output_dir", type=str, default="afs/temp")
    parser.add_argument("--model_name_or_path", type=str, default="")

    parser.add_argument("--rewrite", type=str, default="no")
    parser.add_argument("--table", type=str, default="no")
    parser.add_argument("--mask", type=str, default="yes")
    parser.add_argument("--frozen", type=str, default="no")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0)
    parser.add_argument("--seed", type=int, default=10)

    parser_args = parser.parse_args()
    print(parser_args)
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    trainingArguments = {
        "e2e": e2eTrainingArguments,
        "numericNLG": numericNLGTrainingArguments,
        "numericNLG_cleaned": numericNLGTrainingArguments,
        "wikibio": wikibioTrainingArguments,
        "numericNLG": numericNLGTrainingArguments,
        "Totto": TottoTrainingArguments,
        "webnlg": webnlgTrainingArguments,
    }
    model_args, data_args, training_args = trainingArguments[parser_args.dataset](rewrite=parser_args.rewrite, output_dir=parser_args.output_dir)
    training_args.per_device_train_batch_size = min(training_args.per_device_train_batch_size, parser_args.batch_size)
    training_args.learning_rate = parser_args.learning_rate if parser_args.learning_rate !=0 else training_args.learning_rate

    set_seed(parser_args.seed)
    if len(parser_args.model_name_or_path) == 0:
        backbones = {
            "gpt2": "afs/gpt2",
            "bart": "afs/facebook/bart-large-cnn",
            "t5": "afs/t5-base",
        }
        parser_args.model_name_or_path = backbones[parser_args.model_type]

    model_class, tokenizer_class = MODEL_CLASSES[parser_args.model_type]
    tokenizer = tokenizer_class.from_pretrained(parser_args.model_name_or_path)
    model = model_class.from_pretrained(parser_args.model_name_or_path)

    if len(tokenizer) <= 50257 and parser_args.model_type == "gpt2":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'bos_token': '[BOS]'})
        model.resize_token_embeddings(len(tokenizer))
        model.set_table_word_embedding()

    train_dataset_params = {
        "tokenizer": tokenizer, "src_path": parser_args.train_src, "tgt_path": parser_args.train_gold, 
        "block_size": tokenizer.model_max_length, "bos_tok": tokenizer.bos_token, "eos_tok": tokenizer.eos_token, 
        "model_type": parser_args.model_type, "dataset": parser_args.dataset}
    eval_dataset_params = {
        "tokenizer": tokenizer, "src_path": parser_args.eval_src, "tgt_path":parser_args.eval_gold,   
        "block_size":tokenizer.model_max_length, "bos_tok":tokenizer.bos_token, "eos_tok":tokenizer.eos_token,
        "model_type": parser_args.model_type, "dataset": parser_args.dataset}

    if parser_args.rewrite == "yes":
        model.set_loss_weight(training_args.mle_weight, training_args.rank_weight)
        model.score_mode()
        train_dataset_params.update({
            "rewrite": "yes",
            "cand_path": parser_args.cand_train_path, "cand_metrics_path": parser_args.cand_train_metrics_path
        })
        eval_dataset_params.update({
            "rewrite": "yes",
            "cand_path": parser_args.cand_eval_path, "cand_metrics_path": parser_args.cand_eval_metrics_path
        })

    if parser_args.table == "yes":
        train_dataset_params.update({
                "table": "yes",
                "mask": parser_args.mask,
                "table_path": parser_args.train_table
        })
        eval_dataset_params.update({
            "table": "yes",
            "mask": parser_args.mask,
            "table_path": parser_args.eval_table
        })
        model.table_mode()
        

    if parser_args.frozen == 'yes': # adaptertune
        for param in model.base_model.parameters():
            param.requires_grad = False
    

    train_dataset = Data2TextTrainDataset(**train_dataset_params)
    eval_dataset = Data2TextTrainDataset(**eval_dataset_params)

    data_collator = DataCollatorForData2TextTrain(tokenizer=tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()
    trainer.save_model()

    results = {}
    eval_output = trainer.evaluate(eval_dataset)
    perplexity = eval_output["eval_loss"]
    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    results.update(result)
    print(results)

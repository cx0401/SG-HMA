from ast import arg
from numpy import dtype
import torch
from transformers import set_seed, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
import logging
import os
from arguments import *
import argparse
from data_process.config import mkdir_files
from table_dataloader import Data2TextGenerationDataset, DataCollatorForData2TextGeneration
from torch.utils.data import DataLoader
from tqdm import tqdm

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--fp16",action="store_true",help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/bart-large-cnn", help="")
    parser.add_argument("--table_path", type=str, default="afs/webnlg/data/rewrite0/test/table.json", help="")
    parser.add_argument("--table", type=str, default="no", help="")
    parser.add_argument("--mask", type=str, default="yes", help="")
    parser.add_argument("--curr_dir", type=str, default="temp.txt", help="")
    parser.add_argument("--src_dir", type=str, default="afs/webnlg/data/rewrite0/test/src.txt", help="")
    parser.add_argument("--model_type", type=str, default="bart", help="")
    parser.add_argument("--num_return_sequences", type=int, default=0, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--num_beams", type=int, default=0, help="")
    parser.add_argument("--gen_can", type=str, default="no", help="")
    parser.add_argument("--dataset", type=str, default="webnlg", help="")
    args = parser.parse_args()
    mkdir_files(args.curr_dir)

    generationArguments = {
        "e2e": e2eGenerationArguments,
        "numericNLG": numericNLGGenerationArguments,
        "numericNLG_cleaned": numericNLGGenerationArguments,
        "wikibio": wikibioGenerationArguments,
        "Totto": numericNLGGenerationArguments,
        "webnlg": webnlgGenerationArguments,
    }
    generationArguments = generationArguments[args.dataset]()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.num_beams = generationArguments.num_beams if args.num_beams == 0 else args.num_beams
    args.num_return_sequences = args.num_return_sequences if args.num_return_sequences > 0 else generationArguments.num_return_sequences
    args.batch_size = min(args.batch_size, generationArguments.batch_size)
    print(args)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO,)
    logger = logging.getLogger(__name__)
    logger.warning("device: %s, n_gpu: %s, 16-bits training: %s",args.device,args.n_gpu,args.fp16,)
    set_seed(generationArguments.seed)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    
    if args.model_type=='gpt2':
        tokenizer.padding_side = "left"
    model = model_class.from_pretrained(args.model_name_or_path, ignore_mismatched_sizes=True).to(args.device)
    model.generation_mode()

    if args.fp16:
        model.half()
    print(args)

    out_handle = open(args.curr_dir, 'w')

    test_dataset_params = {
        "tokenizer": tokenizer, "src_path": args.src_dir,
        "block_size": tokenizer.model_max_length, "bos_tok": tokenizer.bos_token, "eos_tok": tokenizer.eos_token,"model_type": args.model_type}
    if args.table == "yes":
        model.table_mode()
        test_dataset_params.update({
            "table": args.table,
            "mask": args.mask,
            "table_path": args.table_path,
        })


    test_dataset = Data2TextGenerationDataset(**test_dataset_params)
    data_collator = DataCollatorForData2TextGeneration(tokenizer=tokenizer, model_type=args.model_type)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    

    with tqdm(dataloader) as bar:
        for (i, batch) in enumerate(bar):
            batch = {key: batch[key].to(args.device) for key in batch}
            # if args.gen_can == "yes":
            #     batch.update({
            #         "diversity_penalty": generationArguments.diversity_penalty,
            #         "no_repeat_ngram_size": generationArguments.no_repeat_ngram_size,
            #         "num_beam_groups": args.num_beams,
            #     })
            
            if args.model_type=='gpt2':
                max_length=generationArguments.max_length + batch["input_ids"].shape[1]
                min_length=generationArguments.min_length + batch["input_ids"].shape[1]
                batch.update({
                "bos_token_id":tokenizer.bos_token_id,
                })
            else:
                max_length = generationArguments.max_length
                min_length = generationArguments.min_length

            batch.update({
                "max_length":max_length,
                "min_length":min_length,
                "temperature": generationArguments.temperature,
                "top_k": generationArguments.k,
                "top_p": generationArguments.p,
                "eos_token_id":tokenizer.eos_token_id,
                "pad_token_id":tokenizer.pad_token_id,
                "repetition_penalty":generationArguments.repetition_penalty,
                "do_sample":generationArguments.do_sample,
                "num_beams":args.num_beams,
                "num_return_sequences": args.num_return_sequences,
            })

            output_sequences = model.generate(**batch)
            generated_sequences = []

            for generated_sequence_idx, src_generated_sequence in enumerate(output_sequences):
                # print("=== GENERATED SEQUENCE {}===".format(prompt_idx + generated_sequence_idx))
                if args.model_type=='gpt2':
                    separator = tokenizer.bos_token_id
                    sep_idx = src_generated_sequence.tolist().index(separator)
                    src_generated_sequence = src_generated_sequence.tolist()[sep_idx + 1:]
                text = tokenizer.decode(src_generated_sequence, clean_up_tokenization_spaces=True)
                if args.model_type in ["gpt2", "bart"]:
                    text = text.replace(tokenizer.bos_token, "")
                text_output = text.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
                if len(text_output) == 0:
                    text_output = "<empty>"
                out_handle.write(text_output + "\n")
                if generated_sequence_idx % args.num_return_sequences == args.num_return_sequences - 1:
                    out_handle.write("\n")
                out_handle.flush()
            
            inputs = []
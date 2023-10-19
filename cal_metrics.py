import argparse
import json
import sys
from numpy import argmin, argmax
import os
import torch
from data_process.config import mkdir_files
from e2e_metrics.metrics.pymteval import BLEUScore, NISTScore
from e2e_metrics.pycocotools.coco import COCO
from e2e_metrics.pycocoevalcap.eval import COCOEvalCap
from nlgeval import NLGEval
# from bert_score import score

def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out

def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out

def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval

def read_file(file):
    refs = [[]]
    with open(file, 'r') as f:
        for (i, line) in enumerate(f):
            if args.lower:
                line = line.lower()
            if line == "\n":
                refs.append([])
            else:
                # refs[-1].append(line.replace('\n', '').replace('.', '').strip())
                refs[-1].append(line.replace("\n", "").replace("[BOS]", "").strip())
    refs.pop()
    return refs

def cal_bleu(hyps, refs):
    metrics = {'BLEU':[]}
    for index, ref in enumerate(refs):
        metrics["BLEU"].append([])
        hyp = hyps[index]
        for h in hyp:
            bleu = BLEUScore()
            bleu.append(h, ref)
            metrics["BLEU"][-1].append(bleu.score())
    return metrics

# def score(srcs, hyps, refs, args):
#     # srcs = srcs[:10]
#     # hyps = hyps[:10]
#     # refs = refs[:10]
#     import torch
#     from transformers import AutoTokenizer
#     from arguments import MODEL_CLASSES
#     import copy
#     device = "cuda"
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
#     tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
#     model = model_class.from_pretrained(args.model_name_or_path).to(device)
#     model.eval()
#     if len(tokenizer) <= 50257:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     cand_inputs = [[src + refs[i][0]] + [src + hyp for hyp in hyps[i]] for i, src in enumerate(srcs)]
#     cand_input_ids = [tokenizer(cand_input, padding=True)["input_ids"] for cand_input in cand_inputs]
#     cands_mask = copy.deepcopy(cand_input_ids)
#     for i, cand_input in enumerate(cand_input_ids):
#         for j, elem in enumerate(cand_input):
#             sep_idx = elem.index(tokenizer.bos_token_id)
#             pad_idx = elem.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in elem else len(elem)
#             cands_mask[i][j] = [0] * (sep_idx + 1) + [1] * (pad_idx - sep_idx - 2) + [0] * (len(elem) - pad_idx + 1)

#     scores = []
#     with torch.no_grad():
#         for i in range(len(hyps)):
#             scores.append(model.score(torch.tensor(cand_input_ids[i:i+1]).to(device), torch.tensor(cands_mask[i:i+1]).to(device)))
#     return scores

def score(args):
    import torch
    from transformers import AutoTokenizer
    from arguments import MODEL_CLASSES
    from table_dataloader import Data2TextEvaluationDataset, DataCollatorForData2TextEvaluation
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    device = "cuda"

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path).to(device)
    model.eval()
    data_collator = DataCollatorForData2TextEvaluation(tokenizer=tokenizer)
    eval_dataset_params = {
        "tokenizer": tokenizer, "src_path": args.src, 
        "block_size":tokenizer.model_max_length, "bos_tok":tokenizer.bos_token, "eos_tok":tokenizer.eos_token, 
        "rewrite": "yes",
        "cand_path": args.hyp,
        "table": "yes",
        "mask": args.mask,
        "table_path": args.table_path,
        # "tgt_path":args.ref,   
        # "cand_metrics_path": args.out_every_path
    }
    eval_dataset = Data2TextEvaluationDataset(**eval_dataset_params)
    dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
    scores = []
    with torch.no_grad():
        with tqdm(dataloader) as bar:
            for batch in bar:
                batch = {key:batch[key].to(device) for key in batch}
                score = model.score(**batch).detach().cpu()
                scores.append(score)
    return scores

def bleuScore(hyps, refs):
    bleu = BLEUScore()
    for index, (sents_ref, sent_sys) in enumerate(zip(refs, hyps)):
        bleu.append(sent_sys, sents_ref)
    return bleu.score()

def nistScore(hyps, refs):
    nist = NISTScore()
    for index, (sents_ref, sent_sys) in enumerate(zip(refs, hyps)):
        nist.append(sent_sys, sents_ref)
    return nist.score()

def getMetrics(refs, hyps, index_choose):
    hyps_choose = [hyp[index_choose[i]] for i, hyp in enumerate(hyps)]
    # hyps_choose = [hyp[0] for i, hyp in enumerate(hyps)]
    average_metrics = {}
    bleu = BLEUScore()
    nist = NISTScore()
    for index, (sents_ref, sent_sys) in enumerate(zip(refs, hyps_choose)):
        bleu.append(sent_sys, sents_ref)
        nist.append(sent_sys, sents_ref)
    average_metrics["BLEU"] = bleu.score()
    try:
        average_metrics["NIST"] = nist.score()
    except Exception as e:
        average_metrics["NIST"] = 0
        print("!!!!!!!!!!!error:", e)
    coco_eval = run_coco_eval(refs, [h[0] for h in hyps])
    average_metrics["ROUGE_L"] = coco_eval.eval["ROUGE_L"]
    average_metrics["METEOR"] = coco_eval.eval["METEOR"]
    average_metrics["CIDEr"] = coco_eval.eval["CIDEr"]
    return average_metrics

def getBLEU(hyps, refs):
    metrics = {'BLEU':[]}
    for index, ref in enumerate(refs):
        metrics["BLEU"].append([])
        hyp = hyps[index]
        for h in hyp:
            bleu = BLEUScore()
            bleu.append(h, ref)
            metrics["BLEU"][-1].append(bleu.score())
    metrics = cal_bleu(hyps, refs)
    return metrics

def getNLGEval(refs, hyps, index_choose):
    nlg_eval = NLGEval(no_skipthoughts=True, no_glove=True)
    hyps_choose = [hyp[index_choose[i]] for i, hyp in enumerate(hyps)]
    # hyps_choose = [hyp[0] for i, hyp in enumerate(hyps)]
    metrics_dict = {'Bleu_1': [], 'Bleu_2': [], 'Bleu_3': [], 'Bleu_4': [], 'ROUGE_L': [], "METEOR":[], "CIDEr": []}
    for i in range(len(refs)):
        ref = refs[i]
        hyp = hyps_choose[i]
        metrics = nlg_eval.compute_individual_metrics(ref, hyp)
        for key in metrics:
            metrics_dict[key].append(metrics[key])
    
    average_metric = {key:sum(metrics_dict[key]) * 100 / len(hyps) for key in metrics_dict}
    return average_metric

def getBertScores(refs, hyps):
    all_bert_score = []
    for i in zip(*sum(all_candidates,[])):
        all_bert_score.append(score(list(i), references * len(all_candidates), lang="en", verbose=True, device="cuda")[2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--src", type=str, default="afs/numericNLG/data/gpt2_rewrite0_table/test/src.txt")
    parser.add_argument("--ref", type=str, default="afs/numericNLG/data/gpt2_rewrite0_table/test/gold.txt")
    parser.add_argument("--hyp", type=str, default="afs/numericNLG/result/gpt2_rewrite0_table/test/out.txt")
    parser.add_argument("--table_path", type=str, default="afs/numericNLG/data/gpt2_rewrite0_table/test/table.json")
    parser.add_argument("--score", type=str, default="no")
    parser.add_argument("--mask", type=str, default="yes")
    parser.add_argument("--model_name_or_path", type=str, default="afs/e2e/checkpoint/rewrite1_table_norewrite")
    parser.add_argument("--model_type", type=str, default="gpt2")
    parser.add_argument("--out_every_path", type=str, default="temp.json")
    parser.add_argument("--out_log_path", type=str, default="afs/e2e/metrics/rewrite0_table/temp.json")
    parser.add_argument("-p", "--python", action="store_true", help="use python rouge")
    parser.add_argument("-l", "--lower", action="store_true", help="lowercase")
    args = parser.parse_args()
    print(args)
    mkdir_files(args.out_every_path)
    mkdir_files(args.out_log_path)

    srcs = [i.replace("\n", "") for i in open(args.src, "r")]
    refs = read_file(args.ref)
    hyps = read_file(args.hyp)

    assert len(hyps) % len(refs) == 0

    index_choose = [0 for _ in range(len(hyps))]

    if args.out_every_path != "":
        metrics = getBLEU(hyps, refs)
        # metric_prefix = getBLEU([[i] for i in open("prefix_out.txt", "r")], refs)
        json.dump(metrics, open(args.out_every_path, "w"))
        real_index_choose = [argmax(metric) for metric in metrics["BLEU"]]
        # index_choose = real_index_choose
    
    if args.score == "yes":
        scores = score(args)
        index_choose = torch.argmax(torch.cat(scores, dim=0), dim=1).tolist()
        print(index_choose)
    
    average_metrics = getNLGEval(refs, hyps, index_choose)
    average_metrics.update(getMetrics(refs, hyps, index_choose))
    print(average_metrics)
    json.dump(average_metrics, open(args.out_log_path, "w"))
        
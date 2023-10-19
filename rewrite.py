import argparse
from data_process.config import mkdir_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_data_dir", default="afs/e2e/data/rewrite0/train/src.txt")
    parser.add_argument("--out_data_dir", default="afs/e2e/result/rewrite0/train/out.txt")
    parser.add_argument("--rewrite_data_dir", default="afs/e2e/data/rewrite1/train/src.txt")
    args = parser.parse_args()
    print(args)
    mkdir_files(args.rewrite_data_dir)
    origin_data = open(args.origin_data_dir, "r").readlines()
    out_data = [[]]
    for line in open(args.out_data_dir, "r"):
        if line == "\n":
            out_data.append([])
        else:
            out_data[-1].append(line)
    out_data.pop()
    assert len(out_data) == len(origin_data)
    with open(args.rewrite_data_dir, "w") as w:
        for i, _ in enumerate(out_data):
            w.write(origin_data[i][:-1].replace("<|endoftext|>", "")+ " . " + out_data[i][0].replace("\n", "") + "\n")

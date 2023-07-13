import json
import os
import random
import numpy as np
import torch as tr
from datetime import datetime
import pandas as pd
import shutil

from torch.utils.data import DataLoader
from sincfold.dataset import SeqDataset, pad_batch
from sincfold.model import sincfold
from sincfold.utils import bp2dot, validate_file
from sincfold.parser import parser

def main():
    
    args = parser()
    
    if not args.no_cache and args.command == "train":
        cache_path = "cache/"
    else:
        cache_path = None

    config= {"device": args.d, "batch_size": args.batch,  "use_restrictions": args.use_restrictions, 
             "valid_split": 0.1, "max_len": 512, "verbose": not args.quiet, "cache_path": cache_path}
    
    if "max_epochs" in args:
        config["max_epochs"] = args.max_epochs

    if args.config is not None:
        config.update(json.load(open(args.config)))

    if config["cache_path"] is not None:
        shutil.rmtree(cache_path, ignore_errors=True)
        os.makedirs(cache_path)

    # Reproducibility
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    def log(msg, file=None):
        if not args.quiet:
            print(msg)
        if file is not None:
            file.write(msg + "\n")
            file.flush()

    if args.command == "train":
        if args.out_path is None:
            out_path = f"results_{str(datetime.today()).replace(' ', '-')}/"
        else:
            out_path = args.out_path
        if out_path[-1] != "/":
            out_path += "/"
        print("working on", out_path)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        else:
            raise ValueError(f"Output path {out_path} already exists")

        if args.valid_file is not None:
            train_file = args.train_file
            valid_file = args.valid_file
        else:
            valid_split = config["valid_split"]
            train_file = f"{out_path}train.csv"
            valid_file = f"{out_path}valid.csv"

            data = pd.read_csv(args.train_file)
            val_data = data.sample(frac = valid_split)
            val_data.to_csv(valid_file, index=False)
            data.drop(val_data.index).to_csv(train_file, index=False)
            
        train_loader = DataLoader(
            SeqDataset(train_file, **config),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=args.j,
            collate_fn=pad_batch,
        )
        valid_loader = DataLoader(
            SeqDataset(valid_file, **config),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=args.j,
            collate_fn=pad_batch,
        )

        net = sincfold(**config)
        
        best_f1, patience_counter = -1, 0
        log("Start training...")
        for epoch in range(config["max_epochs"]):
            train_metrics = net.fit(train_loader)

            val_metrics = net.test(valid_loader)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                tr.save(net.state_dict(), f"{out_path}weights.pmt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 30:
                    break
            msg = (
                f"epoch {epoch}:"
                + " ".join([f"train_{k} {v:.3f}" for k, v in train_metrics.items()])
                + " "
                + " ".join([f"val_{k} {v:.3f}" for k, v in val_metrics.items()])
            )
            log(msg, open(f"{out_path}train.txt", "a"))
        shutil.rmtree(config["cache_path"], ignore_errors=True)


    if args.command == "test":
        test_file = args.test_file

        test_file = validate_file(test_file)

        test_loader = DataLoader(
            SeqDataset(test_file, **config),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=args.j,
            collate_fn=pad_batch,
        )

        if args.model_weights is not None:
            weights = args.model_weights
            net = sincfold(weights=weights, **config)
        else:
            net = sincfold(pretrained=True, **config)
        
        log(f"Start test of {test_file}")        
        test_metrics = net.test(test_loader)
        summary = " ".join([f"test_{k} {v:.3f}" for k, v in test_metrics.items()])
        if args.output_file is not None:
            log(summary, open(args.output_file, "w"))
        else:
            log(summary)

    if args.command == "pred":
        pred_file = args.pred_file
        out_file = args.output_file
        
        pred_file = validate_file(pred_file)
       
        pred_loader = DataLoader(
            SeqDataset(pred_file, max_len=config["max_len"], for_prediction=True, use_restrictions=args.use_restrictions),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=args.j,
            collate_fn=pad_batch,
        )
        
        if args.model_weights is not None:
            weights = args.model_weights
            net = sincfold(weights=weights, **config)
        else:
            net = sincfold(pretrained=True, **config)
                
        log(f"Start prediction of {pred_file}")
        predictions = net.pred(pred_loader)
        
        _, ext = os.path.splitext(out_file)
        if ext == ".fasta":
            with open(out_file, "w") as f:
                for i in range(len(predictions)):
                    item = predictions.iloc[i]
                    structure = bp2dot(len(item.sequence), item.base_pairs)
                    f.write(f">{item.id}\n{structure}\n")
        else:
            predictions.to_csv(out_file, index=False)
                
        
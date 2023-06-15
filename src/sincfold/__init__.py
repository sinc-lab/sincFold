import json
import os
import argparse
import random
import numpy as np
import torch as tr
from datetime import datetime
import pandas as pd

from torch.utils.data import DataLoader
from sincfold.dataset import SeqDataset, pad_batch
from sincfold.model import sincfold
from sincfold.utils import bp2dot, validate_file

def main():
    parser = argparse.ArgumentParser(
        prog="sincFold",
        description="sincFold: An end-to-end method for RNA secondary structure prediction based on deep learning",
        epilog="webserver link | https://github.com/sinc-lab/sincFold",
    )
    parser.add_argument("-c", type=str, help="Config file (optional, overrides any other options)")
    parser.add_argument("-d", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("-batch", type=int, default=4, help="Batch size for handling sequences")
    parser.add_argument("-j", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode") 
    

    subparsers = parser.add_subparsers(
        title="Actions", dest="command", description="Running commands", required=True
    )

    parser_train = subparsers.add_parser("train", help="Train a new model")
    parser_train.add_argument(
        "train_file",
        type=str,
        
        help="Training dataset (csv file with 'id', 'sequence', and 'dotbracket' or 'base_pairs' columns)",
    )

    parser_train.add_argument(
        "--valid_file",
        type=str,
        help="Validation dataset to stop training. If not provided, validation split is randomly generated from training data. Columns are the same as training",
    )
    parser_train.add_argument(
        "-o",
        type=str,
        dest="out_path",
        help="Output path (if not provided, it is generated with the current date)",
    )

    parser_train.add_argument(
        "-n", "--max-epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache of data for training (default: cache is used)",
    )

    parser_test = subparsers.add_parser("test", help="Test a model")
    parser_test.add_argument(
        "test_file",
        type=str,
        help="Testing dataset (csv file with 'id', 'sequence', and 'dotbracket' or 'base_pairs' columns)",
    )
    parser_test.add_argument(
        "-m", type=str, dest="model_weights", help="Trained model weights"
    )

    parser_pred = subparsers.add_parser(
        "pred", help="Predict structures for a list of sequences"
    )
    parser_pred.add_argument(
        "pred_file",
        type=str,
        help="Dataset to predict. It can be a csv file with 'id' and 'sequence' columns or a fasta file",
    )
    parser_pred.add_argument(
        "-o",
        type=str, dest="output_file", default="pred.csv",
        help="Output predictions (default: pred.csv)",
    )
    parser_pred.add_argument(
        "-m", type=str, dest="model_weights", help="Trained model weights"
    )

    args = parser.parse_args()

    config= {"device": args.d, "batch_size": args.batch,  
             "valid_split": 0.1, "max_len": 512, "verbose": not args.quiet}
    if "max_epochs" in args:
        config["max_epochs"] = args.max_epochs

    if args.c is not None:
        config.update(json.load(open(args.config)))

    # Reproducibility
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if args.cache and args.command == "train":
        cache_path = "cache/"
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
    else:
        cache_path = None

    if args.command == "train":
        if args.out_path is None:
            out_path = f"results_{str(datetime.today()).replace(' ', '-')}/"
        else:
            out_path = args.out_path
        if out_path[-1] != "/":
            out_path += "/"
        os.makedirs(out_path)

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
            SeqDataset(train_file, max_len=config["max_len"], cache=cache_path),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=args.j,
            collate_fn=pad_batch,
        )
        valid_loader = DataLoader(
            SeqDataset(valid_file, max_len=config["max_len"], cache=cache_path),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=args.j,
            collate_fn=pad_batch,
        )

        net = sincfold(**config)
        
        best_f1, patience_counter = -1, 0
        print("Start training...")
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
            print(msg)

    if args.command == "test":
        test_file = args.test_file

        test_file = validate_file(test_file)

        test_loader = DataLoader(
            SeqDataset(test_file, max_len=config["max_len"], cache=cache_path),
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
        
        print(f"Start test of {test_file}")
        test_metrics = net.test(test_loader)
        summary = " ".join([f"test_{k} {v:.3f}" for k, v in test_metrics.items()])
        print(summary)

    if args.command == "pred":
        pred_file = args.pred_file
        out_file = args.output_file
        
        pred_file = validate_file(pred_file)
       
        pred_loader = DataLoader(
            SeqDataset(pred_file, max_len=config["max_len"],for_prediction=True),
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
                
        print(f"Start prediction of {pred_file}")
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
                
        
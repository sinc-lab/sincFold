import json
import os
import random
import numpy as np
import torch as tr
from datetime import datetime
import pandas as pd
import shutil
import pickle

from torch.utils.data import DataLoader
from sincfold.dataset import SeqDataset, pad_batch
from sincfold.model import sincfold
from sincfold.embeddings import NT_DICT
from sincfold.utils import write_ct, validate_file, ct2dot
from sincfold.parser import parser
from sincfold.utils import dot2png, ct2svg

def main():
    
    args = parser()
    
    if not args.no_cache and args.command == "train":
        cache_path = "cache/"
    else:
        cache_path = None

    config= {"device": args.d, "batch_size": args.batch, 
             "valid_split": 0.1, "max_len": args.max_length, "verbose": not args.quiet, "cache_path": cache_path}
    
    if "max_epochs" in args:
        config["max_epochs"] = args.max_epochs

    if args.config is not None:
        config.update(json.load(open(args.config)))

    if config["cache_path"] is not None:
        shutil.rmtree(config["cache_path"], ignore_errors=True)
        os.makedirs(config["cache_path"])

    # Reproducibility
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if args.command == "train": 
        train(args.train_file, config, args.out_path,  args.valid_file, args.j)

    if args.command == "test":
        test(args.test_file, args.model_weights, args.out_path, config, args.j)

    if args.command == "pred":
        pred(args.pred_file, model_weights=args.model_weights, out_path=args.out_path, logits=args.logits, config=config, nworkers=args.j, draw=args.draw, draw_resolution=args.draw_resolution)    
        
def train(train_file, config={}, out_path=None, valid_file=None, nworkers=2, verbose=True):
    
    
    if out_path is None:
        out_path = f"results_{str(datetime.today()).replace(' ', '-')}/"
    else:
        out_path = out_path

    if verbose:
        print("Working on", out_path)

    if "cache_path" not in config:
        config["cache_path"] = "cache/"
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    else:
        raise ValueError(f"Output path {out_path} already exists")

    if valid_file is not None:
        train_file = train_file
        valid_file = valid_file
    else:
        data = pd.read_csv(train_file)
        valid_split = config["valid_split"] if "valid_split" in config else 0.1
        train_file = os.path.join(out_path, "train.csv")
        valid_file = os.path.join(out_path, "valid.csv")

        val_data = data.sample(frac = valid_split)
        val_data.to_csv(valid_file, index=False)
        data.drop(val_data.index).to_csv(train_file, index=False)
        
    batch_size = config["batch_size"] if "batch_size" in config else 4
    train_loader = DataLoader(
        SeqDataset(train_file, training=True, **config),
        batch_size=batch_size, 
        shuffle=True,
        num_workers=nworkers,
        collate_fn=pad_batch
    )
    valid_loader = DataLoader(
        SeqDataset(valid_file, **config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )

    net = sincfold(train_len=len(train_loader), **config)
    
    best_f1, patience_counter = -1, 0
    patience = config["patience"] if "patience" in config else 30
    if verbose:
        print("Start training...")
    max_epochs = config["max_epochs"] if "max_epochs" in config else 1000
    logfile = os.path.join(out_path, "train_log.csv") 
        
    for epoch in range(max_epochs):
        train_metrics = net.fit(train_loader)

        val_metrics = net.test(valid_loader)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            tr.save(net.state_dict(), os.path.join(out_path, "weights.pmt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                break
        
        if not os.path.exists(logfile):
            with open(logfile, "w") as f: 
                msg = ','.join(['epoch']+[f"train_{k}" for k in sorted(train_metrics.keys())]+[f"valid_{k}" for k in sorted(val_metrics.keys())]) + "\n"
                f.write(msg)
                f.flush()
                if verbose:
                    print(msg)

        with open(logfile, "a") as f: 
            msg = ','.join([str(epoch)]+[f'{train_metrics[k]:.4f}' for k in sorted(train_metrics.keys())]+[f'{val_metrics[k]:.4f}' for k in sorted(val_metrics.keys())]) + "\n"
            f.write(msg)
            f.flush()    
            if verbose:
                print(msg)
            
    # remove temporal files           
    shutil.rmtree(config["cache_path"], ignore_errors=True)
    
    tmp_file = os.path.join(out_path, "train.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    tmp_file = os.path.join(out_path, "valid.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    
def test(test_file, model_weights=None, output_file=None, config={}, nworkers=2, verbose=True):
    test_file = test_file
    test_file = validate_file(test_file)
    if verbose not in config:
        config["verbose"] = verbose

    test_loader = DataLoader(
        SeqDataset(test_file, **config),
        batch_size=config["batch_size"] if "batch_size" in config else 4,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )

    if model_weights is not None:
        net = sincfold(weights=model_weights, **config)
    else:
        net = sincfold(pretrained=True, **config)
    
    if verbose:
        print(f"Start test of {test_file}")        
    test_metrics = net.test(test_loader)
    summary = ",".join([k for k in sorted(test_metrics.keys())]) + "\n" + ",".join([f"{test_metrics[k]:.3f}" for k in sorted(test_metrics.keys())])+ "\n" 
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(summary)
    if verbose:
        print(summary)

def pred(pred_input, sequence_id='pred_id', model_weights=None, out_path=None, logits=False, config={}, nworkers=2, draw=False, draw_resolution=10, verbose=True):
    
    if out_path is None:
        output_format = "text"
    else:
        _, ext = os.path.splitext(out_path)
        if ext == "":
            if os.path.isdir(out_path):
                raise ValueError(f"Output path {out_path} already exists")
            os.makedirs(out_path)
            output_format = "ct"
        elif ext != ".csv":
            raise ValueError(f"Output path must be a .csv file or a folder, not {ext}")
        else:
            output_format = "csv"

    file_input = os.path.isfile(pred_input)
    if file_input:
        pred_file = validate_file(pred_input)
    else:
        pred_input = pred_input.upper().strip()
        nt_set = set([i for item  in list(NT_DICT.values()) for i in item] + list(NT_DICT.keys()))
        if set(pred_input).issubset(nt_set):
            pred_file = f"{sequence_id}.csv"
            with open(pred_file, "w") as f:
                f.write("id,sequence\n")
                f.write(f"{sequence_id},{pred_input}\n")
            
        else:
            raise ValueError(f"Invalid input nt {set(pred_input)}, either the file is missing or the secuence have invalid nucleotides (should be any of {nt_set})")
    pred_loader = DataLoader(
        SeqDataset(pred_file, for_prediction=True, **config),
        batch_size=config["batch_size"] if "batch_size" in config else 4,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )
    
    if model_weights is not None:
        weights = model_weights
        net = sincfold(weights=weights, **config)
    else:
        net = sincfold(pretrained=True, **config)

    if verbose:        
        print(f"Start prediction of {pred_file}")

    predictions, logits_list = net.pred(pred_loader, logits=logits)
    if draw:
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            ctfile = "tmp.ct"
            write_ct(ctfile, item.id, item.sequence, item.base_pairs)
            dotbracket = ct2dot(ctfile)
            
            png_file = item.id +".png"
            if out_path is not None and os.path.isdir(out_path):
                png_file = os.path.join(out_path, png_file)
            if dotbracket:
                dot2png(png_file, item.sequence, dotbracket, resolution=draw_resolution)
            ct2svg("tmp.ct", png_file.replace(".png", ".svg"))

    if not file_input:
        os.remove(pred_file)

    if output_format == "text":
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            ctfile = "tmp.ct"
            write_ct(ctfile, item.id, item.sequence, item.base_pairs)
            dotbracket = ct2dot(ctfile)
            print(item.id)
            print(item.sequence)
            print(dotbracket)
            print()
    elif output_format == "csv":
        predictions.to_csv(out_path, index=False)
    else: # ct
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            write_ct(os.path.join(out_path, item.id +".ct"), item.id, item.sequence, item.base_pairs)
    if logits:
        base = os.path.split(out_path)[0] if not os.path.isdir(out_path) else out_path
        if len(base) == 0:
            base = "."
        out_path_dir = base + "/logits/"
        os.mkdir(out_path_dir)
        for id, pred, pred_post in logits_list:
            pickle.dump((pred, pred_post), open(os.path.join(out_path_dir, id + ".pk"), "wb"))

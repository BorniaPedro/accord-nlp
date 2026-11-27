import argparse
import itertools
import json
import os
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def parse_args():
    parser = argparse.ArgumentParser(description="Orquestrador GLiNER-ReLEx (Grid Search completo ou Modelos Finais).")
    parser.add_argument("--model_path", type=str, default="knowledgator/gliner-relex-large-v0.5")
    parser.add_argument("--final_only", action="store_true", help="Treina e avalia apenas os 2 modelos vencedores da Tabela 1.")

    # Espaço de busca completo (216 combinações)
    parser.add_argument("--epochs_list", nargs="+", default=["10", "15"])
    parser.add_argument("--batch_sizes", nargs="+", default=["16"])
    parser.add_argument("--lrs", nargs="+", default=["1e-5", "2e-5", "3e-5"])
    parser.add_argument("--others_lrs", nargs="+", default=["5e-5", "1e-4"])
    parser.add_argument("--ner_thresholds", nargs="+", default=["0.30", "0.35", "0.40"])
    parser.add_argument("--rel_thresholds", nargs="+", default=["0.40", "0.50", "0.60"])

    parser.add_argument("--scenarios", nargs="+", default=["standard", "augmented"], choices=["standard", "augmented"])
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument("--drive_backup", type=str, default=None)
    return parser.parse_args()


def run_final_winners(args):
    """Reproduz exatamente os resultados da Tabela 4 e salva test_predictions.json para inspeção manual."""
    winners = [
        {
            "key": "standard",
            "name": "Standard_Winner_Val",
            "train_path": "data/relex/relex_train.json",
            "out_dir": "outputs/final_models/standard",
            "epochs": "15", "lr": "3e-5", "others_lr": "1e-4",
            "ner_th": "0.40", "rel_th": "0.50", "aug": False
        },
        {
            "key": "augmented",
            "name": "Augmented_Winner_Val",
            "train_path": "data/relex/relex_train_augmented.json",
            "out_dir": "outputs/final_models/augmented",
            "epochs": "10", "lr": "3e-5", "others_lr": "5e-5",
            "ner_th": "0.40", "rel_th": "0.60", "aug": True
        }
    ]
    winners = [w for w in winners if w["key"] in args.scenarios]
    consolidated_report = {}

    for cfg in winners:
        print(f"\n{'='*80}\nExecutando Modelo Final: {cfg['name']}\n{'='*80}")
        cmd = [
            "python", "-m", "experiments.relex.train_relex",
            "--model_path", args.model_path,
            "--train_path", cfg["train_path"],
            "--output_dir", cfg["out_dir"],
            "--csv_out", "outputs/final_models/summary_winners.csv",
            "--epochs", cfg["epochs"],
            "--batch_size", args.batch_sizes[0],
            "--grad_accum", str(args.grad_accum),
            "--fp16",
            "--lr", cfg["lr"],
            "--others_lr", cfg["others_lr"],
            "--ner_thresholds", cfg["ner_th"],
            "--rel_thresholds", cfg["rel_th"],
            "--run_name", cfg["name"],
            "--save_predictions"
        ]
        if cfg["aug"]:
            cmd.append("--augmented_data")

        subprocess.run(cmd, check=True)

        report_file = Path(cfg["out_dir"]) / "per_class_report.json"
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                consolidated_report[cfg["name"]] = json.load(f)

    out_summary = Path("outputs/per_class_results_seed42.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(consolidated_report, f, indent=2, ensure_ascii=False)
    print(f"\nRelatório consolidado (Tabela 4) salvo em: {out_summary.resolve()}")


def main():
    args = parse_args()
    total_start = time.time()

    if args.final_only:
        run_final_winners(args)
        print(f"\nConcluído em: {format_duration(time.time() - total_start)}")
        return

    scenario_configs = {
        "standard": {"train_path": "data/relex/relex_train.json", "augmented_flag": False, "prefix": "Standard"},
        "augmented": {"train_path": "data/relex/relex_train_augmented.json", "augmented_flag": True, "prefix": "Augmented"}
    }

    train_grid = list(itertools.product(args.epochs_list, args.batch_sizes, args.lrs, args.others_lrs))
    total_runs = len(args.scenarios) * len(train_grid)
    current_run = 0

    print("=" * 85)
    print(f"GRID SEARCH COMPLETO GLINER-RELEX")
    print(f"Treinos: {total_runs} | Combinações com thresholds: {total_runs * len(args.ner_thresholds) * len(args.rel_thresholds)}")
    print("=" * 85)

    csv_local = Path("outputs") / "benchmark_results_full_grid.csv"

    for sc_key in args.scenarios:
        sc = scenario_configs[sc_key]
        for ep, bs, lr, oth_lr in train_grid:
            current_run += 1
            run_name = f"{sc['prefix']}_ep{ep}_bs{bs}_lr{lr}_oth{oth_lr}"
            out_dir = os.path.join("outputs", "grid_relex_full", sc_key, f"ep{ep}_bs{bs}_lr{lr}_oth{oth_lr}")

            cmd = [
                "python", "-m", "experiments.relex.train_relex",
                "--model_path", args.model_path,
                "--train_path", sc["train_path"],
                "--output_dir", out_dir,
                "--epochs", str(ep),
                "--batch_size", str(bs),
                "--grad_accum", str(args.grad_accum),
                "--fp16",
                "--lr", str(lr),
                "--others_lr", str(oth_lr),
                "--ner_thresholds", *args.ner_thresholds,
                "--rel_thresholds", *args.rel_thresholds,
                "--run_name", run_name
            ]
            if sc["augmented_flag"]:
                cmd.append("--augmented_data")
            if not args.keep_checkpoints:
                cmd.append("--no_save_ckpt")

            print(f"\n[{current_run}/{total_runs}] Treinando: {run_name}")
            t0 = time.time()
            proc = subprocess.run(cmd)
            status = "OK" if proc.returncode == 0 else "ERRO"
            print(f"-> Status: {status} | Tempo: {format_duration(time.time() - t0)}")

            if csv_local.exists() and args.drive_backup:
                shutil.copy(csv_local, args.drive_backup)

    print(f"\nGRID SEARCH FINALIZADO EM: {format_duration(time.time() - total_start)}")

    if csv_local.exists():
        df = pd.read_csv(csv_local)
        for aug_val, label in [(False, "STANDARD"), (True, "AUGMENTED")]:
            sub = df[df["Augmented_Data"] == aug_val]
            if len(sub) > 0:
                best_row = sub.sort_values(by="Val_RE_Strict_Macro_F1", ascending=False).iloc[0]
                print(f"\n>>> VENCEDOR NA VALIDAÇÃO - {label}:\n{best_row.to_string()}")


if __name__ == "__main__":
    main()
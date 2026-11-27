# experiments/ner/run_grid.py
import argparse
import os
import subprocess
import time
from datetime import timedelta


def parse_args():
    parser = argparse.ArgumentParser(description="Grid Search GLiNER com medicao de tempo")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Numero de epocas por combinacao")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Tamanho do batch por dispositivo (16 otimizado para GPU)")
    parser.add_argument("--output_base", type=str, default="outputs/grid",
                        help="Diretorio base de saida")
    parser.add_argument("--models", nargs="+", default=[
        "gliner-community/gliner_small-v2.5",
        "urchade/gliner_medium-v2.1"
    ], help="Modelos a avaliar (small e medium)")
    parser.add_argument("--max_widths", nargs="+", type=int, default=[12, 20, 32],
                        help="Lista de valores para max_width")
    parser.add_argument("--focal_gammas", nargs="+", type=float, default=[0.0, 2.0],
                        help="Lista de valores para focal_loss_gamma")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def main():
    args = parse_args()
    total_grid_start = time.time()
    results_summary = []

    total_combinations = len(args.models) * len(args.max_widths) * len(args.focal_gammas)
    current_run = 0

    print("=" * 70)
    print(f"INICIANDO GRID SEARCH GLiNER | Epocas: {args.epochs} | Batch: {args.batch_size}")
    print(f"Total de combinacoes: {total_combinations}")
    print("=" * 70)

    for model in args.models:
        model_slug = model.split("/")[-1]
        
        for width in args.max_widths:
            for gamma in args.focal_gammas:
                current_run += 1
                
                run_tag = f"w{width}_g{gamma}_e{args.epochs}"
                out_dir = os.path.join(args.output_base, model_slug, run_tag)
                alpha_val = "0.5" if gamma > 0.0 else "-1.0"

                cmd = [
                    "python", "-m", "experiments.ner.ner_experiments_gliner",
                    "--model_name", model,
                    "--max_width", str(width),
                    "--focal_loss_gamma", str(gamma),
                    "--focal_loss_alpha", alpha_val,
                    "--num_epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--output_dir", out_dir
                ]

                print(f"\n[{current_run}/{total_combinations}] Executando: {model_slug} | {run_tag}")
                print(f"Destino: {out_dir}")

                run_start = time.time()
                process = subprocess.run(cmd)
                run_duration = time.time() - run_start

                status = "Sucesso" if process.returncode == 0 else "Falhou"
                duration_str = format_duration(run_duration)

                print(f"Status: {status} | Tempo: {duration_str}")

                results_summary.append({
                    "model": model_slug,
                    "max_width": width,
                    "gamma": gamma,
                    "epochs": args.epochs,
                    "duration": duration_str,
                    "status": status,
                    "dir": out_dir
                })

    total_grid_duration = time.time() - total_grid_start

    print("\n" + "=" * 70)
    print("RELATORIO CONSOLIDADO DO GRID SEARCH")
    print(f"Tempo total gasto: {format_duration(total_grid_duration)}")
    print("=" * 70)
    print(f"{'Modelo':<26} | {'Width':<5} | {'Gamma':<5} | {'Epocas':<6} | {'Duracao':<10} | {'Status'}")
    print("-" * 70)

    for item in results_summary:
        print(f"{item['model']:<26} | {item['max_width']:<5} | {item['gamma']:<5} | {item['epochs']:<6} | {item['duration']:<10} | {item['status']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
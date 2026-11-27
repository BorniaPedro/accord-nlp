# data_processing/relex/download_relex_model.py
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import torch


def setup_local_relex_model(
    repo_id="knowledgator/gliner-relex-large-v1.0",
    local_dir="models/gliner_relex_large"
):
    target_path = Path(local_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"BAIXANDO CHECKPOINT BASE: {repo_id}")
    print("=" * 65)

    # 1. Baixa todos os arquivos do repositório oficial
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_path),
        local_dir_use_symlinks=False
    )
    print(f"Arquivos baixados em: {target_path.resolve()}")

    # 2. Garante a existência do pytorch_model.bin para o glirel
    bin_file = target_path / "pytorch_model.bin"
    safetensors_file = target_path / "model.safetensors"

    if not bin_file.exists():
        if safetensors_file.exists():
            print("Convertendo 'model.safetensors' para 'pytorch_model.bin'...")
            tensors = load_file(str(safetensors_file))
            torch.save(tensors, str(bin_file))
            print("Conversão concluída com sucesso!")
        else:
            raise FileNotFoundError("Arquivo de pesos não encontrado no repositório.")
    else:
        print("Arquivo 'pytorch_model.bin' já presente.")

    print("=" * 65)
    print("MODELO PRONTO PARA USO LOCAL!")
    print("=" * 65)


if __name__ == "__main__":
    setup_local_relex_model()
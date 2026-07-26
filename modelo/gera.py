import torch
from pathlib import Path
import torch.nn.functional as F

from modelo import Modelo
from train import carregar_modelo


caminho = Path(__file__).resolve().parent / "pesos" / "modelo.pt"

def gerar_texto(
    modelo,
    tokenizer,
    texto_inicial,
    quantidade_tokens=200,
    temperatura=1.0
):
    dispositivo = next(
        modelo.parameters()
    ).device

    modelo.eval()

    tokens = tokenizer.codificar(
        texto_inicial
    )

    tokens = torch.tensor(
        tokens,
        dtype=torch.long,
        device=dispositivo
    )

    # Adiciona a dimensão do lote:
    # [tokens] -> [1, tokens]
    tokens = tokens.unsqueeze(0)

    with torch.no_grad():
        for _ in range(quantidade_tokens):
            # O modelo aceita no máximo tamanho_bloco tokens.
            contexto = tokens[
                :,
                -modelo.tamanho_bloco:
            ]

            logits, _ = modelo(contexto)

            # Pega somente a previsão da última posição.
            logits_ultimo_token = logits[:, -1, :]

            # Controla a aleatoriedade.
            logits_ultimo_token = (
                logits_ultimo_token / temperatura
            )

            probabilidades = F.softmax(
                logits_ultimo_token,
                dim=-1
            )

            proximo_token = torch.multinomial(
                probabilidades,
                num_samples=1
            )

            tokens = torch.cat(
                [tokens, proximo_token],
                dim=1
            )

    numeros = tokens[0].tolist()

    return tokenizer.decodificar(numeros)

def main():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    modelo, tokenizer = carregar_modelo(
        caminho=caminho,
        dispositivo=dispositivo
    )

    prompt = input("Digite o início do texto: ")

    resultado = gerar_texto(
        modelo=modelo,
        tokenizer=tokenizer,
        texto_inicial=prompt,
        quantidade_tokens=300,
        temperatura=0.8
    )

    print("\nTexto gerado:\n")
    print(resultado)


if __name__ == "__main__":
    main()
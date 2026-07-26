import torch

from modelo import Modelo
from train import carregar_modelo
from train import gerar_texto


def main():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    modelo, tokenizer = carregar_modelo(
        caminho="mini_llm.pth",
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
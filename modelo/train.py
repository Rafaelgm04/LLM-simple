import torch
from tqdm import tqdm
import modelo as M
from pathlib import Path

import tratamento as u

PASTA_ATUAL = Path(__file__).resolve().parent
caminho = PASTA_ATUAL / "pesos" / "modelo.pt"

def salvar_modelo(caminho, modelo, tokenizer, configuracao):
    caminho = Path(caminho)

    # Cria a pasta "pesos" caso ainda não exista
    caminho.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "pesos_modelo": modelo.state_dict(),
        "caracteres": tokenizer.caracteres,
        "configuracao": configuracao
    }

    torch.save(checkpoint, caminho)

    print(f"\nModelo salvo em: {caminho.resolve()}")

def carregar_modelo(caminho, dispositivo):
    checkpoint = torch.load(
        caminho,
        map_location=dispositivo
    )

    configuracao = checkpoint["configuracao"]

    caracteres = checkpoint["caracteres"]

    tokenizer = u.Tokenizer(
        "".join(caracteres)
    )

    modelo = M.Modelo(
        tamanho_vocab=configuracao["tamanho_vocab"],
        tamanho_bloco=configuracao["tamanho_bloco"],
        tamanho_embedding=configuracao["tamanho_embedding"],
        numero_cabecas=configuracao["numero_cabecas"],
        numero_camadas=configuracao["numero_camadas"],
        dropout=configuracao["dropout"]
    )

    modelo.load_state_dict(
        checkpoint["pesos_modelo"]
    )

    modelo = modelo.to(dispositivo)

    modelo.eval()

    return modelo, tokenizer


def calcular_acuracia(logits, respostas):
    previsoes = torch.argmax(
        logits,
        dim=-1
    )

    acertos = previsoes == respostas

    return acertos.float().mean().item()

def train_model(
    modelo,
    tokens_treino,
    n_epochss=1000,
    tamanho_bloco=64,
    tamanho_lote=16,
    taxa_aprendizado=3e-4,
    configuracao=None,
    tokenizer=None


):
    modelo.train()

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=taxa_aprendizado
    )

    dispositivo = next(
        modelo.parameters()
    ).device

    losss = []
    acuracias = []

    barra = tqdm(
        range(n_epochss),
        desc="Treinamento"
    )

    for epochs in barra:
        entradas, respostas = u.cria_lote(
            tokens_treino,
            tamanho_bloco=tamanho_bloco,
            tamanho_lote=tamanho_lote
        )

        entradas = entradas.to(dispositivo)
        respostas = respostas.to(dispositivo)

        logits, loss = modelo(
            entradas,
            respostas
        )

        otimizador.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            modelo.parameters(),
            max_norm=1.0
        )

        otimizador.step()

        acuracia = calcular_acuracia(
            logits,
            respostas
        )
        
        losss.append(loss.item())
        acuracias.append(acuracia)

        if epochs % 100 == 0:
            barra.set_postfix(
                loss=f"{loss.item():.4f}",
                acuracia=f"{acuracia:.2%}"
            )

        print("epochs: ",epochs)
        print("loss: ",f"{loss.item():.4f}")

        if configuracao is not None and tokenizer is not None:
            if (epochs + 1) % 1000 == 0:
                salvar_modelo(
                    caminho=caminho,
                    modelo=modelo,
                    tokenizer=tokenizer,
                    configuracao=configuracao
                )
        

    loss_media = sum(losss) / len(losss)
    acuracia_media = sum(acuracias) / len(acuracias)

    if configuracao is not None and tokenizer is not None:
        salvar_modelo(
            caminho=caminho,
            modelo=modelo,
            tokenizer=tokenizer,
            configuracao=configuracao
        )

    return loss_media, acuracia_media

def run_epoch(
    modelo,
    tokens_validacao,
    quantidade_lotes=100,
    tamanho_bloco=64,
    tamanho_lote=16
):
    modelo.eval()

    dispositivo = next(
        modelo.parameters()
    ).device

    losss = []
    acuracias = []

    with torch.no_grad():
        for _ in tqdm(
            range(quantidade_lotes),
            desc="Validação"
        ):
            entradas, respostas = u.cria_lote(
                tokens_validacao,
                tamanho_bloco=tamanho_bloco,
                tamanho_lote=tamanho_lote
            )

            entradas = entradas.to(dispositivo)
            respostas = respostas.to(dispositivo)

            logits, loss = modelo(
                entradas,
                respostas
            )

            acuracia = calcular_acuracia(
                logits,
                respostas
            )

            losss.append(loss.item())
            acuracias.append(acuracia)

    loss_media = sum(losss) / len(losss)
    acuracia_media = sum(acuracias) / len(acuracias)

    return loss_media, acuracia_media
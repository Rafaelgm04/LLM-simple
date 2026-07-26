import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

import tratamento as u

import train as train
CAMINHO_DATASET = Path("../Datasets")


class BlocoTransformer(nn.Module):
    def __init__(
        self,
        tamanho_embedding,
        numero_cabecas,
        dropout
    ):

        super().__init__()

        self.normalizacao_1 = nn.LayerNorm(
            tamanho_embedding
        )

        self.atencao = nn.MultiheadAttention(
            embed_dim=tamanho_embedding,
            num_heads=numero_cabecas,
            dropout=dropout,
            batch_first=True
        )

        self.normalizacao_2 = nn.LayerNorm(
            tamanho_embedding
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                tamanho_embedding,
                4 * tamanho_embedding
            ),
            nn.GELU(),
            nn.Linear(
                4 * tamanho_embedding,
                tamanho_embedding
            ),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tamanho_sequencia = x.shape[1]

        mascara_causal = torch.triu(
            torch.ones(
                tamanho_sequencia,
                tamanho_sequencia,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x_normalizado = self.normalizacao_1(x)

        resultado_atencao, _ = self.atencao(
            x_normalizado,
            x_normalizado,
            x_normalizado,
            attn_mask=mascara_causal,
            need_weights=False
        )

        x = x + self.dropout(resultado_atencao)

        x = x + self.feed_forward(
            self.normalizacao_2(x)
        )

        return x

        


class Modelo(nn.Module):
    def __init__(self, 
                 tamanho_vocab,
                 tamanho_bloco=64,
                 tamanho_embedding=128,
                 numero_cabecas=4,
                 numero_camadas=4,
                 dropout=0.1):
        
        super().__init__()

        self.tamanho_bloco     = tamanho_bloco

        self.embedding_token   = nn.Embedding(
            tamanho_vocab, 
            tamanho_embedding)
        
        self.embedding_posicao = nn.Embedding(
            tamanho_bloco, 
            tamanho_embedding)


        self.blocos = nn.Sequential(
            *[
                BlocoTransformer(
                    tamanho_embedding,
                    numero_cabecas,
                    dropout
                )
                for _ in range(numero_camadas)

            ]
        )

        self.normalizacao_final = nn.LayerNorm(
            tamanho_embedding
        )

        self.cabeca_saida = nn.Linear(
            tamanho_embedding,
            tamanho_vocab
        )


    def forward(self, tokens, respostas=None):
        quantidade_lotes, tamanho_sequencia = tokens.shape

        embedding_tokens = self.embedding_token(tokens)

        if tamanho_sequencia > self.tamanho_bloco:
            raise ValueError(
                "A sequência é maior que o tamanho do bloco."
            )
        
        embedding_tokens = self.embedding_token(tokens)

        posicoes = torch.arange(
            tamanho_sequencia,
            device=tokens.device
        ) 

        embedding_posicoes = self.embedding_posicao(posicoes)

        x = embedding_tokens + embedding_posicoes

        x = self.blocos(x)

        x = self.normalizacao_final(x)

        logits = self.cabeca_saida(x)

        perda = None

        if respostas is not None:
            perda = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.shape[-1]
                ),
                respostas.reshape(-1)
            )

        return logits, perda


def main():

    caminho_treino = CAMINHO_DATASET / "treino.txt"
    caminho_validacao = CAMINHO_DATASET / "validacao.txt"

    with open(caminho_treino,"r", encoding="utf-8") as arquivo:
        texto_treino = arquivo.read()

    with open(caminho_validacao,"r", encoding="utf-8") as arquivo:
        texto_validacao = arquivo.read()

    tokenizer = u.Tokenizer(texto_treino + texto_validacao)

    tokens_train = torch.tensor(tokenizer.codificar(texto_treino), dtype=torch.long)
    tokens_test  = torch.tensor(tokenizer.codificar(texto_validacao), dtype=torch.long)


    configuracao = {
        "tamanho_vocab": tokenizer.tamanho_vocab,
        "tamanho_bloco": 64,
        "tamanho_embedding": 128,
        "numero_cabecas": 4,
        "numero_camadas": 4,
        "dropout": 0.1
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tamanho_vocab = len(
        tokenizer.char_para_id
    )

    rede = Modelo(
        tamanho_vocab=tamanho_vocab,
        tamanho_bloco=64,
        tamanho_embedding=128,
        numero_cabecas=4,
        numero_camadas=4,
        dropout=0.1
    )

    rede = rede.to(device)


    while True:
            print("\n==========================")
            print("       MENU DO MODELO")
            print("==========================")
            print("1 - Treinar modelo")
            print("2 - Ver loss")
            print("3 - Testar modelo")
            print("0 - Sair")
    
            opcao = input("\nEscolha uma opção: ").strip()

            
            if opcao == "1":
                n_epochss = input("n_epochss")
                loss_train, acc_train = train.train_model(rede,tokens_train,n_epochss=n_epochss,configuracao=configuracao,tokenizer=tokenizer)
                print("\nTreinamento concluído")
    
            elif opcao == "2":
                print(
                    f"Perda média treino: {loss_train:.4f}"
                )
                
                print(
                    f"Acurácia média: {acc_train:.2%}"
                )

                print("\nResultado da validação")
                print(

                    f"Perda: {loss_test:.4f}"
                )

                print(
                    f"Acurácia: {acc_test:.2%}" 
                )

    
            elif opcao == "3":
                loss_test, acc_test = train.run_epoch(rede, tokens_test)
    
            elif opcao == "0":
                print("\nEncerrando o programa...")
                break
    
            else:
                print("\nOpção inválida. Escolha 0, 1, 2 ou 3.")
    




if __name__ == "__main__":
    #torch.manual_seed(12321)
    main()



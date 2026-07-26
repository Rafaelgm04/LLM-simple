import torch

class Tokenizer:
    def __init__(self, texto):
        self.texto = texto
        caracteres = sorted(set(texto))

        self.tamanho_vocab = len(caracteres)
        self.caracteres = caracteres

        self.char_para_id  = {
            caractere: indice
            for indice, caractere in enumerate(caracteres)
        }

        self.id_para_char  = {
            indice: caractere
            for caractere, indice in self.char_para_id.items()
        }

    def codificar(self, texto):
        return [self.char_para_id[c] for c in texto]

    def decodificar(self, numeros):
        return "".join(self.id_para_char[n] for n in numeros)


def cria_lote(dados, tamanho_bloco=64, tamanho_lote=16):

    if len(dados) <= tamanho_bloco:
        raise ValueError(
            "O texto é menor que o tamanho do bloco."
        )

    posicoes = torch.randint(
        0,
        len(dados) - tamanho_bloco - 1,
        (tamanho_lote,)
    )

    entradas = torch.stack([
        dados[posicao:posicao + tamanho_bloco]
        for posicao in posicoes
    ])

    respostas = torch.stack([
        dados[posicao + 1:posicao + tamanho_bloco + 1]
        for posicao in posicoes
    ])

    return entradas, respostas


        

        
from datasets import load_dataset
import hashlib

dataset = load_dataset(
    "TucanoBR/wikipedia-PT",
    split="train",
    streaming=True
)

limite_total = 20_000_000  # aproximadamente 20 milhões de caracteres

caracteres_treino = 0
caracteres_validacao = 0
caracteres_total = 0

with open("treino.txt", "w", encoding="utf-8") as treino, \
     open("validacao.txt", "w", encoding="utf-8") as validacao:

    for exemplo in dataset:
        texto = exemplo["text"].strip()

        if not texto:
            continue

        # Cria um número estável baseado no conteúdo do artigo
        hash_texto = hashlib.sha256(
            texto.encode("utf-8")
        ).hexdigest()

        numero = int(hash_texto[:8], 16)

        # Aproximadamente 10% vai para validação
        if numero % 10 == 0:
            validacao.write(texto + "\n\n")
            caracteres_validacao += len(texto)
        else:
            treino.write(texto + "\n\n")
            caracteres_treino += len(texto)

        caracteres_total += len(texto)

        if caracteres_total >= limite_total:
            break

print(f"Treino: {caracteres_treino:,} caracteres")
print(f"Validação: {caracteres_validacao:,} caracteres")
print(f"Total: {caracteres_total:,} caracteres")
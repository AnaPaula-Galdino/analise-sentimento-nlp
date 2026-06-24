"""
Gera avaliações de clientes em português (texto + nota + sentimento).
Inclui avaliações MISTAS e termos ambíguos para refletir a dificuldade real
da tarefa de NLP (evitando um problema separável demais). Autora: Ana Paula Galdino
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(53)

sujeitos = ["o produto", "a entrega", "o atendimento", "a embalagem", "a compra", "o pedido"]
pos_adj = ["excelente","ótimo","maravilhoso","perfeito","incrível","muito bom","impecável","bacana"]
pos_frase = ["superou as expectativas","chegou antes do prazo","recomendo","voltarei a comprar",
             "boa qualidade","valeu o preço","entrega rápida","atendimento atencioso"]
neg_adj = ["péssimo","horrível","ruim","decepcionante","fraco","frustrante"]
neg_frase = ["chegou quebrado","demorou para chegar","não recomendo","veio com defeito",
             "abaixo do esperado","tive problema na troca","poderia ser melhor"]
neu_frase = ["é ok","razoável pelo preço","esperava mais mas serve","cumpre o que promete",
             "mediano","atende ao básico","dentro do esperado","comum, sem destaque"]
# clausulas mistas (deixam o texto ambíguo, mantendo o rótulo geral)
mix_em_pos = ["mas a entrega demorou um pouco","só achei meio caro","apesar de um arranhão na caixa"]
mix_em_neg = ["mas o atendimento tentou ajudar","ainda assim a embalagem era bonita","porém chegou rápido"]

def review(sent):
    s = rng.choice(sujeitos)
    if sent == "positivo":
        partes = [f"{s} é {rng.choice(pos_adj)}", rng.choice(pos_frase)]
        if rng.random() < 0.20: partes.append(rng.choice(mix_em_pos))
        nota = int(rng.choice([4,5,3], p=[0.4,0.5,0.1]))
    elif sent == "negativo":
        partes = [f"{s} é {rng.choice(neg_adj)}", rng.choice(neg_frase)]
        if rng.random() < 0.20: partes.append(rng.choice(mix_em_neg))
        nota = int(rng.choice([1,2,3], p=[0.45,0.45,0.10]))
    else:
        partes = [rng.choice(neu_frase)]
        # neutro às vezes empresta um termo leve de pos/neg -> ambiguidade
        if rng.random() < 0.5: partes.append(rng.choice(neu_frase))
        if rng.random() < 0.25: partes.append(rng.choice(pos_frase + neg_frase))
        nota = int(rng.choice([3,2,4], p=[0.7,0.15,0.15]))
    rng.shuffle(partes)
    return (", ".join(partes).capitalize() + "."), nota

def nota_do_rotulo(sent):
    if sent=="positivo": return int(rng.choice([4,5],p=[0.4,0.6]))
    if sent=="negativo": return int(rng.choice([1,2],p=[0.5,0.5]))
    return 3

linhas=[]
dist={"positivo":0.55,"negativo":0.28,"neutro":0.17}
for _ in range(2500):
    sent=rng.choice(list(dist), p=list(dist.values()))
    # ~12% de avaliacoes ambiguas: texto de outro sentimento (ruido real de rotulagem)
    texto_sent = sent
    if rng.random() < 0.12:
        texto_sent = rng.choice([c for c in dist if c != sent])
    t,_=review(texto_sent)
    linhas.append([t, nota_do_rotulo(sent), sent])
df=pd.DataFrame(linhas, columns=["avaliacao","nota","sentimento"])
df.to_csv(os.path.join(BASE,"dados","avaliacoes.csv"), index=False)
print(f"Avaliações: {len(df)}"); print(df["sentimento"].value_counts().to_dict())

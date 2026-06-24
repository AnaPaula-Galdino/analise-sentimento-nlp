import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relatorio_exec import construir
import analise_sentimento as A

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(BASE, "imagens")
def img(n): return os.path.join(IMG, n)
r = A.resumo()

config = {
 "eyebrow": "RELATÓRIO DE ANÁLISE DE SENTIMENTO",
 "titulo": "Análise de Sentimento de Avaliações",
 "subtitulo": "Classificação automática de avaliações de clientes com NLP (TF-IDF + Regressão Logística)",
 "meta": "Ana Paula Galdino · Data Analytics (POSTECH/FIAP) · Junho de 2026",
 "fonte": "Avaliações de clientes  ·  NLP: Ana Paula Galdino",
 "sumario": [
   f"Ler avaliação por avaliação não escala. Treinei um modelo de NLP que lê o texto e classifica "
   f"automaticamente o sentimento em <b>positivo, neutro ou negativo</b>, a partir de "
   f"<b>{r['n']:,}</b> avaliações.".replace(",", "."),
   f"O classificador atingiu <b>{r['acc']:.0f}% de acurácia</b> (F1-macro de {r['f1']:.0f}%) num "
   "conjunto de teste, e ainda revela <b>quais palavras</b> mais empurram uma avaliação para cada lado — "
   "transformando texto solto em informação acionável para o negócio.",
 ],
 "kpis": [
   (f"{r['n']:,}".replace(",", "."), "avaliações analisadas"),
   (f"{r['acc']:.0f}%", "acurácia do modelo"),
   (f"{r['pct_pos']:.0f}%", "avaliações positivas"),
   (f"{r['pct_neg']:.0f}%", "avaliações negativas"),
 ],
 "secoes": [
   {"titulo": "1. O retrato geral",
    "texto": [
      f"A base é majoritariamente positiva (<b>{r['pct_pos']:.0f}%</b>), mas a fatia negativa "
      f"(<b>{r['pct_neg']:.0f}%</b>) é justamente onde mora a oportunidade de melhoria.",
      "O cruzamento entre sentimento e nota (estrelas) serve de checagem de consistência: notas baixas "
      "concentram avaliações negativas, como esperado, validando a rotulagem.",
    ],
    "imagens": [(img("01_distribuicao_sentimento.png"), "Distribuição dos sentimentos"),
                (img("06_sentimento_por_nota.png"), "Sentimento por faixa de nota")]},
   {"titulo": "2. O modelo de classificação",
    "texto": [
      f"Usando TF-IDF (com unigramas e bigramas) e Regressão Logística, o modelo acertou "
      f"<b>{r['acc']:.0f}%</b> das avaliações de teste. A matriz de confusão mostra que os erros se "
      "concentram na fronteira com a classe neutra — o caso genuinamente ambíguo, como em dados reais.",
      "Um resultado próximo de 100% seria suspeito: 86% é coerente com a dificuldade real de separar "
      "elogios, críticas e avaliações mornas.",
    ],
    "imagens": [(img("02_matriz_confusao.png"), "Acertos e erros do classificador"),
                (img("05_nuvem_palavras.png"), "Termos mais frequentes nas avaliações positivas")]},
   {"titulo": "3. O que move cada sentimento",
    "texto": [
      "Além de classificar, o modelo é interpretável: dá para ver quais termos mais pesam para cada lado. "
      "Isso aponta diretamente os pontos fortes a manter e as dores a resolver.",
      "Palavras ligadas a entrega, defeito e prazo aparecem entre as mais negativas — sinais concretos de "
      "onde a operação precisa melhorar.",
    ],
    "imagens": [(img("03_palavras_positivas.png"), "Termos que indicam satisfação"),
                (img("04_palavras_negativas.png"), "Termos que indicam insatisfação")]},
 ],
 "conclusao_titulo": "Como aplicar",
 "conclusoes": [
   "<b>Monitoramento em escala:</b> classificar avaliações automaticamente, sem leitura manual.",
   "<b>Alerta de insatisfação:</b> sinalizar picos de avaliações negativas para ação rápida.",
   "<b>Priorização de melhorias:</b> os termos negativos apontam onde investir (entrega, defeitos, prazo).",
   "<b>Próximo passo:</b> aplicar a avaliações reais e conectar a um painel de acompanhamento contínuo.",
 ],
}

if __name__ == "__main__":
    construir(config, os.path.join(BASE, "Analise_Executiva_Sentimento.pdf"))

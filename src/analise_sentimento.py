"""
Análise de Sentimento de Avaliações — NLP
Classifica avaliações em positivo / neutro / negativo (TF-IDF + Regressão Logística)
e extrai os termos que mais pesam em cada sentimento.
Autora: Ana Paula Galdino
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from wordcloud import WordCloud

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(BASE, "imagens"); os.makedirs(IMG, exist_ok=True)
C = {"escuro":"#1f4e79","medio":"#2e6da4","claro":"#5b9bd5","suave":"#a6c8e0",
     "destaque":"#4fc3f7","cinza":"#d9d9d9","alerta":"#c0392b","verde":"#2e7d32"}
FONTE = "Avaliações de clientes  ·  NLP: Ana Paula Galdino"
STOP = ["a","o","os","as","de","do","da","dos","das","e","é","que","com","um","uma","no","na",
        "em","para","por","mas","muito","mais","ao","se","sem","pelo","nada","demais","ser","esse"]
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":"#9aa7b8",
    "axes.grid":True,"grid.color":"#eef2f7","axes.axisbelow":True,"figure.dpi":120,"savefig.bbox":"tight"})
def rodape(fig): fig.text(0.01,0.005,FONTE,fontsize=7.5,color="#7a8aa0")

df = pd.read_csv(os.path.join(BASE,"dados","avaliacoes.csv"))
classes = ["negativo","neutro","positivo"]
Xtr,Xte,ytr,yte = train_test_split(df["avaliacao"], df["sentimento"], test_size=0.25,
                                   random_state=42, stratify=df["sentimento"])
pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words=STOP, ngram_range=(1,2), min_df=2)),
                 ("clf", LogisticRegression(max_iter=1000, C=4))]).fit(Xtr, ytr)
pred = pipe.predict(Xte)
acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="macro")

vec = pipe.named_steps["tfidf"]; clf = pipe.named_steps["clf"]
vocab = np.array(vec.get_feature_names_out())
def top_termos(classe, n=12):
    idx = list(clf.classes_).index(classe)
    coef = clf.coef_[idx]
    top = np.argsort(coef)[-n:]
    return vocab[top], coef[top]

# 1) Distribuição de sentimento
def g1():
    cont = df["sentimento"].value_counts().reindex(classes)
    cores=[C["alerta"],C["cinza"],C["verde"]]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(cont.index, cont.values, color=cores)
    for i,v in enumerate(cont.values): ax.text(i,v,f"{v}\n({v/len(df)*100:.0f}%)",ha="center",va="bottom",fontsize=10)
    ax.set_title("Distribuição dos sentimentos nas avaliações",fontweight="bold",color=C["escuro"],fontsize=13,pad=10)
    ax.set_ylabel("Avaliações"); ax.set_ylim(0,cont.max()*1.18)
    rodape(fig); fig.savefig(os.path.join(IMG,"01_distribuicao_sentimento.png")); plt.close(fig)

# 2) Matriz de confusão
def g2():
    cm = confusion_matrix(yte, pred, labels=classes)
    fig,ax=plt.subplots(figsize=(6,5.2)); im=ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=13,
                    color="white" if cm[i,j]>cm.max()/2 else "#1f4e79")
    ax.set_title(f"Matriz de confusão — acurácia {acc*100:.0f}%",fontweight="bold",color=C["escuro"],fontsize=13,pad=10)
    ax.grid(False)
    rodape(fig); fig.savefig(os.path.join(IMG,"02_matriz_confusao.png")); plt.close(fig)

# 3) Palavras positivas
def g3():
    w,c = top_termos("positivo")
    fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.barh(w, c, color=C["verde"])
    ax.set_title("Termos que mais indicam avaliação POSITIVA",fontweight="bold",color=C["escuro"],fontsize=13,pad=10)
    ax.set_xlabel("Peso no modelo")
    rodape(fig); fig.savefig(os.path.join(IMG,"03_palavras_positivas.png")); plt.close(fig)

# 4) Palavras negativas
def g4():
    w,c = top_termos("negativo")
    fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.barh(w, c, color=C["alerta"])
    ax.set_title("Termos que mais indicam avaliação NEGATIVA",fontweight="bold",color=C["escuro"],fontsize=13,pad=10)
    ax.set_xlabel("Peso no modelo")
    rodape(fig); fig.savefig(os.path.join(IMG,"04_palavras_negativas.png")); plt.close(fig)

# 5) Nuvem de palavras
def g5():
    texto = " ".join(df[df["sentimento"]=="positivo"]["avaliacao"].tolist())
    wc = WordCloud(width=1000, height=500, background_color="white",
                   colormap="Blues", stopwords=set(STOP), min_font_size=10).generate(texto)
    fig,ax=plt.subplots(figsize=(10,5)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Nuvem de palavras — avaliações positivas",fontweight="bold",color=C["escuro"],fontsize=13,pad=10)
    rodape(fig); fig.savefig(os.path.join(IMG,"05_nuvem_palavras.png")); plt.close(fig)

# 6) Sentimento por nota
def g6():
    tab = pd.crosstab(df["nota"], df["sentimento"], normalize="index")[classes]*100
    fig,ax=plt.subplots(figsize=(9,5)); bottom=np.zeros(len(tab))
    for cls,cor in zip(classes,[C["alerta"],C["cinza"],C["verde"]]):
        ax.bar(tab.index.astype(str), tab[cls], bottom=bottom, color=cor, label=cls)
        bottom += tab[cls].values
    ax.set_title("Sentimento por nota (estrelas)",fontweight="bold",color=C["escuro"],fontsize=13,pad=10)
    ax.set_xlabel("Nota"); ax.set_ylabel("% das avaliações"); ax.legend(title="Sentimento", frameon=True)
    rodape(fig); fig.savefig(os.path.join(IMG,"06_sentimento_por_nota.png")); plt.close(fig)

def resumo():
    cont=df["sentimento"].value_counts()
    return {"n":len(df),"acc":acc*100,"f1":f1*100,
            "pct_pos":cont["positivo"]/len(df)*100,"pct_neg":cont["negativo"]/len(df)*100,
            "top_neg": list(top_termos("negativo",5)[0][::-1])}

def main():
    for g in (g1,g2,g3,g4,g5,g6): g()
    r=resumo()
    print(f"Acurácia={acc*100:.1f}% F1-macro={f1*100:.1f}%")
    print("Top termos negativos:", r["top_neg"])
    print("Gráficos:", sorted(x for x in os.listdir(IMG) if x.startswith("0")))
    return r

if __name__=="__main__":
    main()

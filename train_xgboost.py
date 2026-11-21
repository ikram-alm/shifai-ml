# %% [markdown]
# # ShifAI — Entraînement XGBoost (NHANES 2013-2023)
#
# Objectif : prédire l'insulinorésistance (HOMA-IR >= 2.5) à partir de
# signaux NON invasifs (âge, IMC, tour de taille, tension, antécédents...).
# Les analyses sanguines (glycémie, insuline) servent UNIQUEMENT à calculer
# la cible : elles ne sont jamais des variables d'entrée du modèle.
#
# Dans VS Code : chaque bloc "# %%" est une cellule exécutable
# (cliquez sur "Run Cell"), ou lancez tout avec : python train_xgboost.py

# %% Imports et configuration
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # enregistre les graphiques en fichiers
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve)

DATA_DIR = Path("data")          # dossier contenant vos fichiers .xpt
OUT_DIR = Path("outputs")        # modèle, graphiques, métriques
OUT_DIR.mkdir(exist_ok=True)

HOMA_SEUIL = 2.5                 # seuil d'insulinorésistance (à justifier dans le rapport)
SENSIBILITE_CIBLE = 0.80         # en dépistage, on veut rater peu de personnes à risque
SEED = 42

# Suffixe NHANES -> cycle
CYCLES = {"H": "2013-2014", "I": "2015-2016", "J": "2017-2018", "L": "2021-2023"}


import sys
print("=== ShifAI train_xgboost.py - VERSION 2 (avec validation temporelle) ===")
print("Python utilise :", sys.executable)


# %% 1. Chargement des fichiers .xpt
def trouver_fichier(prefixe: str, suffixe: str):
    """Cherche ex. GLU_H.xpt / glu_h.XPT n'importe où dans DATA_DIR."""
    cible = f"{prefixe}_{suffixe}.xpt".lower()
    for p in DATA_DIR.rglob("*"):
        if p.name.lower() == cible:
            return p
    return None


def lire(prefixe: str, suffixe: str):
    p = trouver_fichier(prefixe, suffixe)
    if p is None:
        return None
    df = pd.read_sas(p, format="xport")
    df["SEQN"] = df["SEQN"].astype(int)
    return df


def charger_cycle(s: str) -> pd.DataFrame:
    demo = lire("DEMO", s)
    if demo is None:
        raise FileNotFoundError(f"DEMO_{s}.xpt non trouve dans {DATA_DIR}")
    df = demo
    # BPX (manuel) jusqu'en 2017-2018, BPXO (oscillométrique) en 2021-2023
    for prefixe in ["BMX", "BPX", "BPXO", "GLU", "INS", "GHB", "DIQ", "MCQ", "SMQ"]:
        d = lire(prefixe, s)
        if d is None:
            print(f"   (absent) {prefixe}_{s}")
            continue
        cols = [c for c in d.columns if c == "SEQN" or c not in df.columns]
        df = df.merge(d[cols], on="SEQN", how="left")
    df["cycle"] = CYCLES[s]
    return df


frames = []
for s in CYCLES:
    if trouver_fichier("DEMO", s) is None:
        print(f"Cycle {CYCLES[s]} ignore (pas de DEMO_{s}.xpt)")
        continue
    print(f"Chargement {CYCLES[s]} ...")
    frames.append(charger_cycle(s))

if not frames:
    raise FileNotFoundError(
        "Aucun cycle exploitable : aucun fichier DEMO_H/I/J/L.xpt n'a ete trouve "
        f"dans {DATA_DIR}. Telechargez au moins DEMO_H.xpt, DEMO_I.xpt, "
        "DEMO_J.xpt ou DEMO_L.xpt et placez-les dans les dossiers data/"
    )

raw = pd.concat(frames, ignore_index=True)
print("Donnees brutes :", raw.shape)


# %% 2. Harmonisation des variables entre cycles
def premiere_colonne(df, candidats):
    """Renvoie la 1re colonne existante parmi les candidats, ligne par ligne."""
    out = pd.Series(np.nan, index=df.index)
    for c in candidats:
        if c in df.columns:
            out = out.fillna(df[c])
    return out


def moyenne_colonnes(df, candidats):
    cols = [c for c in candidats if c in df.columns]
    return df[cols].mean(axis=1) if cols else pd.Series(np.nan, index=df.index)


def col(nom):
    """Colonne de raw, ou une colonne vide si la variable n'existe pas."""
    return raw[nom] if nom in raw.columns else pd.Series(np.nan, index=raw.index)


d = pd.DataFrame({"SEQN": raw["SEQN"], "cycle": raw["cycle"]})

# --- Labs (pour la CIBLE seulement) ---
d["glucose"] = premiere_colonne(raw, ["LBXGLU"])                 # mg/dL
d["insuline"] = premiere_colonne(raw, ["LBXIN"])                 # µU/mL
d["insuline"] = d["insuline"].fillna(col("LBDINSI") / 6.0)     # pmol/L -> µU/mL
d["hba1c"] = premiere_colonne(raw, ["LBXGH"])

# --- Variables d'entrée NON invasives ---
d["age"] = col("RIDAGEYR")
d["sexe_femme"] = (col("RIAGENDR") == 2).astype(float)
d["origine"] = col("RIDRETH3")                               # catégoriel
d["poids"] = col("BMXWT")
d["taille"] = col("BMXHT")
d["imc"] = col("BMXBMI")
d["tour_taille"] = col("BMXWAIST")
d["ratio_taille_hauteur"] = d["tour_taille"] / d["taille"]
d["tension_sys"] = moyenne_colonnes(raw, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
                                          "BPXOSY1", "BPXOSY2", "BPXOSY3"])
d["tension_dia"] = moyenne_colonnes(raw, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
                                          "BPXODI1", "BPXODI2", "BPXODI3"])
d.loc[d["tension_dia"] <= 0, "tension_dia"] = np.nan             # 0 = valeur invalide
# Antécédents familiaux de diabète (MCQ300C : 1 = oui, 2 = non)
d["antecedent_familial"] = col("MCQ300C").map({1: 1.0, 2: 0.0})
# Tabac (SMQ020 : a fumé >= 100 cigarettes)
d["fumeur"] = col("SMQ020").map({1: 1.0, 2: 0.0})

# --- Exclusions ---
diab_diag = col("DIQ010") == 1          # diabète déjà diagnostiqué
insuline_trt = col("DIQ050") == 1       # sous insuline
pilules = col("DIQ070") == 1            # sous antidiabétiques oraux
enceinte = col("RIDEXPRG") == 1

masque = (
    (d["age"] >= 18)
    & ~diab_diag & ~insuline_trt & ~pilules & ~enceinte
    & d["glucose"].notna() & d["insuline"].notna()
)
d = d[masque].copy()

# --- Cible : HOMA-IR (formule déterministe) ---
d["homa_ir"] = d["glucose"] * d["insuline"] / 405
d["cible"] = (d["homa_ir"] >= HOMA_SEUIL).astype(int)

print("Echantillon final :", len(d))
print(d.groupby("cycle")["cible"].agg(["count", "mean"]).rename(
    columns={"count": "n", "mean": "% insulinoresistants"}))

FEATURES = ["age", "sexe_femme", "origine", "poids", "taille", "imc", "tour_taille",
            "ratio_taille_hauteur", "tension_sys", "tension_dia",
            "antecedent_familial", "fumeur"]
# Retire les colonnes entièrement vides (ex. fichier MCQ non téléchargé)
FEATURES = [f for f in FEATURES if d[f].notna().any()]
print("Variables utilisees :", FEATURES)

X = d[FEATURES].astype(float)
y = d["cible"]


# %% 3. Découpage train / validation / test (stratifié)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=SEED)
print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")


# %% 4. Baseline : régression logistique (pour comparer)
baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))
baseline.fit(X_train, y_train)
p_base = baseline.predict_proba(X_test)[:, 1]
print(f"Baseline logistique  ROC-AUC={roc_auc_score(y_test, p_base):.3f}  "
      f"PR-AUC={average_precision_score(y_test, p_base):.3f}")


# %% 5. XGBoost (gère les valeurs manquantes nativement)
ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=ratio,      # compense le déséquilibre des classes
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
print("Meilleure iteration :", model.best_iteration)


# %% 6. Évaluation sur le test
p_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, p_test)
ap = average_precision_score(y_test, p_test)
print(f"XGBoost  ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

# Seuil choisi sur la VALIDATION pour atteindre la sensibilité cible
p_val = model.predict_proba(X_val)[:, 1]
fpr, tpr, seuils = roc_curve(y_val, p_val)
seuil = float(seuils[np.argmax(tpr >= SENSIBILITE_CIBLE)])

pred = (p_test >= seuil).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metriques = {
    "roc_auc": round(auc, 4),
    "pr_auc": round(ap, 4),
    "baseline_logistic_roc_auc": round(roc_auc_score(y_test, p_base), 4),
    "seuil_decision": round(seuil, 4),
    "sensibilite": round(tp / (tp + fn), 4),
    "specificite": round(tn / (tn + fp), 4),
    "vpp": round(tp / (tp + fp), 4) if tp + fp else None,
    "n_test": int(len(y_test)),
    "homa_seuil": HOMA_SEUIL,
    "features": FEATURES,
}
print(json.dumps(metriques, indent=2, ensure_ascii=True))

# Courbe ROC
fpr_t, tpr_t, _ = roc_curve(y_test, p_test)
fpr_b, tpr_b, _ = roc_curve(y_test, p_base)
plt.figure(figsize=(5, 5))
plt.plot(fpr_t, tpr_t, label=f"XGBoost (AUC {auc:.2f})")
plt.plot(fpr_b, tpr_b, label=f"Logistique (AUC {roc_auc_score(y_test, p_base):.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("1 - specificite"); plt.ylabel("Sensibilite"); plt.legend()
plt.title("Courbe ROC - test")
plt.tight_layout(); plt.savefig(OUT_DIR / "roc.png", dpi=150); plt.close()


# %% 7. Explicabilité (SHAP)
try:
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    shap.summary_plot(sv, X_test, show=False)
    plt.tight_layout(); plt.savefig(OUT_DIR / "shap_summary.png", dpi=150); plt.close()
    print("SHAP enregistre -> outputs/shap_summary.png")
except Exception as e:
    print("SHAP impossible :", repr(e))
    print(f'  -> installez-le avec : "{sys.executable}" -m pip install shap')


# %% 7b. Validation temporelle : entraine sur 2013-2018, teste sur 2021-2023
# Le modele ne voit JAMAIS les donnees 2021-2023 pendant l'entrainement.
# C'est plus proche de la realite : on deploie un modele sur des gens "du futur".
from sklearn.base import clone

CYCLES_TRAIN = ["2013-2014", "2015-2016", "2017-2018"]
CYCLE_TEST = "2021-2023"

passe = d["cycle"].isin(CYCLES_TRAIN).values
futur = (d["cycle"] == CYCLE_TEST).values

if futur.sum() == 0:
    print("Validation temporelle ignoree : pas de donnees 2021-2023")
else:
    Xp, yp = X[passe], y[passe]
    Xf, yf = X[futur], y[futur]
    Xp_tr, Xp_val, yp_tr, yp_val = train_test_split(
        Xp, yp, test_size=0.20, stratify=yp, random_state=SEED)

    # Memes reglages que le modele principal
    model_t = clone(model)
    model_t.set_params(scale_pos_weight=(yp_tr == 0).sum() / (yp_tr == 1).sum())
    model_t.fit(Xp_tr, yp_tr, eval_set=[(Xp_val, yp_val)], verbose=False)

    base_t = clone(baseline).fit(Xp_tr, yp_tr)

    # Seuil fixe sur la validation 2013-2018, puis applique tel quel a 2021-2023
    fpr_v, tpr_v, s_v = roc_curve(yp_val, model_t.predict_proba(Xp_val)[:, 1])
    seuil_t = float(s_v[np.argmax(tpr_v >= SENSIBILITE_CIBLE)])

    pf = model_t.predict_proba(Xf)[:, 1]
    pf_base = base_t.predict_proba(Xf)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yf, (pf >= seuil_t).astype(int)).ravel()

    metriques["validation_temporelle"] = {
        "entrainement": CYCLES_TRAIN,
        "test": CYCLE_TEST,
        "n_train": int(len(Xp_tr)),
        "n_test": int(len(Xf)),
        "roc_auc": round(roc_auc_score(yf, pf), 4),
        "pr_auc": round(average_precision_score(yf, pf), 4),
        "baseline_logistic_roc_auc": round(roc_auc_score(yf, pf_base), 4),
        "seuil_decision": round(seuil_t, 4),
        "sensibilite": round(tp / (tp + fn), 4),
        "specificite": round(tn / (tn + fp), 4),
        "vpp": round(tp / (tp + fp), 4) if tp + fp else None,
    }

    print("\n=== Validation temporelle (train 2013-2018 -> test 2021-2023) ===")
    print(f"{'':28s}{'Aleatoire':>12s}{'Temporelle':>12s}")
    vt = metriques["validation_temporelle"]
    for k, nom in [("roc_auc", "ROC-AUC XGBoost"),
                   ("baseline_logistic_roc_auc", "ROC-AUC logistique"),
                   ("sensibilite", "Sensibilite"),
                   ("specificite", "Specificite"),
                   ("vpp", "VPP")]:
        print(f"{nom:28s}{metriques[k]:>12.3f}{vt[k]:>12.3f}")

    # Courbe ROC comparee
    fa, ta, _ = roc_curve(y_test, p_test)
    ft, tt, _ = roc_curve(yf, pf)
    plt.figure(figsize=(5, 5))
    plt.plot(fa, ta, label=f"Decoupage aleatoire (AUC {metriques['roc_auc']:.2f})")
    plt.plot(ft, tt, label=f"Temporel 2021-2023 (AUC {vt['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("1 - specificite"); plt.ylabel("Sensibilite"); plt.legend()
    plt.title("XGBoost - validation aleatoire vs temporelle")
    plt.tight_layout(); plt.savefig(OUT_DIR / "roc_temporelle.png", dpi=150); plt.close()
    print("Graphique -> outputs/roc_temporelle.png")


# %% 8. Sauvegarde du modèle
model.save_model(OUT_DIR / "shifai_xgb.json")
with open(OUT_DIR / "metriques.json", "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2, ensure_ascii=True)
print("Modele -> outputs/shifai_xgb.json")


# %% 9. Exemple de prédiction pour un nouvel utilisateur
nouveau = pd.DataFrame([{
    "age": 42, "sexe_femme": 1, "origine": np.nan, "poids": 82, "taille": 162,
    "imc": 82 / 1.62**2, "tour_taille": 98, "ratio_taille_hauteur": 98 / 162,
    "tension_sys": 128, "tension_dia": 82, "antecedent_familial": 1, "fumeur": 0,
}])[FEATURES]
risque = model.predict_proba(nouveau)[0, 1]
print(f"Score de risque : {risque:.2f} -> "
      f"{'A RISQUE, bilan conseille' if risque >= seuil else 'risque faible'}")
import base64
import os
import urllib.request
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Modèles IA — Analyse",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
# Le dashboard ne lit plus de fichiers : il interroge l'API (compétence C5)
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_HEADERS = {"X-API-Key": os.getenv("API_KEY")}

PROVIDER_COLORS = {
    "Google": "#4285F4", "DeepSeek": "#FF6B35", "Mistral": "#FA0F00",
    "Zhipu": "#00B4D8", "Alibaba": "#FF9F1C", "xAI": "#111111",
    "NVIDIA": "#76B900", "Meta": "#0668E1", "Moonshot": "#9B5DE5",
    "Anthropic": "#CC785C", "OpenAI": "#10A37F", "MiniMax": "#6C5CE7",
    "Microsoft": "#00A4EF", "EuroLLM": "#3D9970", "Allen AI": "#E84393",
    "NousResearch": "#F7B731", "Cohere": "#D4A017", "Autre": "#AAAAAA",
    # Fournisseurs absents de l'échantillon du notebook, présents dans les données complètes
    "Swiss AI": "#D52B1E", "Liquid AI": "#6E56CF", "AI21 Labs": "#E8590C",
    "Arcee AI": "#0F766E", "01.AI": "#495057", "Jpacifico": "#8B5E3C",
}

LOGO_URLS = {
    "Google":    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Google_Favicon_2025.svg/250px-Google_Favicon_2025.svg.png",
    "DeepSeek":  "https://raw.githubusercontent.com/lobehub/lobe-icons/refs/heads/master/packages/static-png/dark/deepseek-color.png",
    "Mistral":   "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Mistral_AI_logo_%282025%E2%80%93%29.svg/1280px-Mistral_AI_logo_%282025%E2%80%93%29.svg.png",
    "Zhipu":     "https://raw.githubusercontent.com/lobehub/lobe-icons/refs/heads/master/packages/static-png/light/zai.png",
    "Alibaba":   "https://ik.imagekit.io/businesswith/tr:w-200,h-100,cm-pad_resize,dpr-2/logo/alibaba-cloud-logo.png",
    "xAI":       "https://1000logos.net/wp-content/uploads/2023/10/xAI-Logo.png",
    "NVIDIA":    "https://companieslogo.com/img/orig/NVDA-220e1e03.png?t=1722952498",
    "OpenAI":    "https://assets.streamlinehq.com/image/private/w_240,h_240,ar_1/f_auto/v1/icons/technology/openai_1-moa3pqsiii7l4dkheifi8.png/openai_1-gv7rd0u7lcncyfalyjodt.png?_a=DATAiZiuZAA0",
    "Anthropic": "https://images.seeklogo.com/logo-png/55/2/claude-logo-png_seeklogo-554534.png",
    "Meta":      "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Meta_Platforms_Inc._logo_%28cropped%29.svg/3840px-Meta_Platforms_Inc._logo_%28cropped%29.svg.png",
    "Moonshot":  "https://raw.githubusercontent.com/lobehub/lobe-icons/refs/heads/master/packages/static-png/light/moonshot.png",
    "MiniMax":   "https://raw.githubusercontent.com/lobehub/lobe-icons/refs/heads/master/packages/static-png/dark/minimax-color.png",
    "EuroLLM":   "https://eurollm.io/wp-content/themes/wl_euroLLM/assets/images/eurollm_logo.svg",
    "Allen AI":  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Allen_Institute_for_Artificial_Intelligence.svg/960px-Allen_Institute_for_Artificial_Intelligence.svg.png",
}

CHAMPION_COLORS = {
    "qualite":  "#E63946",
    "efficience": "#2A9D8F",
    "token":    "#E9C46A",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_provider(model: str) -> str:
    m = model.lower()
    if m.startswith(("gemini", "gemma")):                          return "Google"
    if m.startswith("deepseek"):                                    return "DeepSeek"
    if m.startswith(("mistral", "magistral", "ministral", "mixtral")): return "Mistral"
    if m.startswith("glm"):                                         return "Zhipu"
    if m.startswith(("qwen", "qwq")):                              return "Alibaba"
    if m.startswith("grok"):                                        return "xAI"
    if "nemotron" in m:                                             return "NVIDIA"
    if m.startswith(("llama", "hermes")):                          return "Meta"
    if m.startswith("kimi"):                                        return "Moonshot"
    if m.startswith("claude"):                                      return "Anthropic"
    if m.startswith(("gpt", "o3", "o4")):                          return "OpenAI"
    if m.startswith("minimax"):                                     return "MiniMax"
    if m.startswith("phi"):                                         return "Microsoft"
    if m.startswith("eurollm"):                                     return "EuroLLM"
    if m.startswith("olmo"):                                        return "Allen AI"
    # Mêmes règles que src/clean.py (ajoutées le 11/10 pour les données complètes)
    if m.startswith(("aya", "c4ai", "command")):                    return "Cohere"
    if m.startswith("apertus"):                                     return "Swiss AI"
    if m.startswith("lfm"):                                         return "Liquid AI"
    if m.startswith("jamba"):                                       return "AI21 Labs"
    if m.startswith("trinity"):                                     return "Arcee AI"
    if m.startswith("yi-"):                                         return "01.AI"
    if m.startswith("chocolatine"):                                 return "Jpacifico"
    return "Autre"


def pareto_frontier(df):
    df_s = df.sort_values("score", ascending=False)
    pareto, min_kwh = [], float("inf")
    for idx, row in df_s.iterrows():
        if row["avg_kwh"] <= min_kwh:
            pareto.append(idx)
            min_kwh = row["avg_kwh"]
    return df.loc[pareto].sort_values("avg_kwh")


def logo_img_tag(provider: str, size: int = 24) -> str:
    url = LOGO_URLS.get(provider)
    if url:
        return f'<img src="{url}" height="{size}" style="vertical-align:middle;object-fit:contain;border-radius:3px;">'
    color = PROVIDER_COLORS.get(provider, "#AAAAAA")
    ini = provider[0] if provider else "?"
    return (
        f'<span style="background:{color};color:white;border-radius:50%;'
        f'width:{size}px;height:{size}px;display:inline-flex;align-items:center;'
        f'justify-content:center;font-size:{size//2}px;font-weight:bold;">{ini}</span>'
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def fetch(endpoint):
    """Demande un tableau à l'API et le renvoie sous forme de DataFrame."""
    response = requests.get(f"{API_URL}{endpoint}", headers=API_HEADERS, timeout=60)
    response.raise_for_status()  # erreur claire si l'API est arrêtée ou si la clé est fausse
    return pd.DataFrame(response.json())


@st.cache_data
def load_data():
    classement = (
        fetch("/ranking")
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    classement["provider"] = classement["model"].apply(get_provider)

    energy_g = (
        fetch("/energy")
        .query("avg_kwh >= 0.00001")
    )
    energy_g["provider"] = energy_g["model"].apply(get_provider)

    ranking_cat = fetch("/categories")
    ranking_cat["provider"] = ranking_cat["model"].apply(get_provider)

    merged = classement.merge(
        energy_g[["model", "avg_kwh", "avg_kwh_per_req", "mwh_per_token"]],
        on="model", how="inner",
    )
    merged["provider"] = merged["model"].apply(get_provider)
    merged["score_par_kwh"]   = merged["score"] / merged["avg_kwh"]
    merged["score_par_token"] = merged["score"] / merged["mwh_per_token"]

    nb_conv          = int(energy_g["nb_conversations"].sum() // 2)
    nb_models_score  = len(classement)
    nb_models_energy = len(energy_g)
    top_model        = classement.iloc[0]

    return classement, energy_g, merged, ranking_cat, nb_conv, nb_models_score, nb_models_energy, top_model


classement, energy_g, merged, ranking_cat, nb_conv, nb_models_score, nb_models_energy, top_model = load_data()

# Affichage en Wh plutôt qu'en kWh : « 0,031 Wh » se lit mieux que « 3.08e-05 kWh » (1 kWh = 1 000 Wh).
for _df in (energy_g, merged, ranking_cat):
    _df["wh_conv"] = _df["avg_kwh"] * 1000
    _df["wh_req"] = _df["avg_kwh_per_req"] * 1000


def fmt_value(value):
    """Écrit une valeur d'énergie sans notation scientifique, avec un nombre de décimales adapté."""
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"



@st.cache_data
def load_measures():
    """Énergie MESURÉE (AI Energy Score) croisée avec l'énergie estimée (ComparIA), et volumes en base."""
    measures = fetch("/estimated-vs-measured")  # tâche text_generation par défaut
    health = requests.get(f"{API_URL}/health", timeout=60).json()  # endpoint public
    return measures, health


measures, health = load_measures()
# Modèles comparables : énergie ComparIA disponible et au moins 200 conversations (mêmes règles que l'analyse)
comparable = measures[measures["comparia_wh_per_req"].notna() & (measures["nb_conversations"] >= 200)].copy()


def _fetch_logo(prov_url):
    prov, url = prov_url
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            img = Image.open(BytesIO(r.read())).convert("RGBA").resize((40, 40), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return prov, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return prov, None


@st.cache_data
def load_logos():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(LOGO_URLS)) as pool:
        results = pool.map(_fetch_logo, LOGO_URLS.items())
    return {prov: b64 for prov, b64 in results if b64}


logo_b64 = load_logos()


# ══════════════════════════════════════════════════════════════════════════════
# TITRE
# ══════════════════════════════════════════════════════════════════════════════
st.title("Comparaison entre différents modèles d'IA")
st.caption("Source : ComparIA (ministère de la Culture) — votes et conversations en duel, lus via l'API du projet. "
           "Textes mis à jour sur la collecte du 12/10/2026 (conversations du 01/10/2024 au 18/05/2026).")

# Les deux encadrés de présentation des sources, côte à côte
_info_left, _info_right = st.columns(2)

with _info_left.expander("ℹ️ Qu'est-ce que ComparIA ?", expanded=False):
    st.markdown("""
    **ComparIA** est une plateforme publique qui invite les utilisateurs à comparer des modèles d'IA en conditions réelles.
    À chaque session, **le même message est envoyé simultanément à deux modèles différents** — un duel.
    L'utilisateur lit les deux réponses, réagit à chacune, puis vote pour celle qu'il préfère.
    Ces données constituent une base de référence pour évaluer et améliorer les modèles d'IA en langues européennes.
    """)

    st.markdown("#### Les 3 bases de données")
    _ca, _cb, _cc = st.columns(3)

    _card_info = (
        "border-radius:14px;padding:20px;text-align:center;height:175px;"
        "display:flex;flex-direction:column;justify-content:center;gap:8px;"
    )
    with _ca:
        st.markdown(
            f'<div style="{_card_info}background:#EEF2FF;border:1px solid #C7D2FE;">'
            '<div style="font-size:22px;font-weight:700;color:#4F46E5;">/conversations</div>'
            '<div style="font-size:13px;color:#555;">Chaque duel : modèles, énergie estimée, tokens, catégories. Le texte des échanges n\'est pas collecté (RGPD)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with _cb:
        st.markdown(
            f'<div style="{_card_info}background:#F0FDF4;border:1px solid #BBF7D0;">'
            '<div style="font-size:22px;font-weight:700;color:#16A34A;">/réactions</div>'
            '<div style="font-size:13px;color:#555;">Les retours sur chaque réponse (utile, créatif, incorrect…) — non utilisés par ce projet</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with _cc:
        st.markdown(
            f'<div style="{_card_info}background:#FEFCE8;border:1px solid #FDE68A;">'
            '<div style="font-size:22px;font-weight:700;color:#CA8A04;">/votes</div>'
            '<div style="font-size:13px;color:#555;">Le choix exprimé en fin de conversation — quel modèle a fourni la meilleure réponse ?</div>'
            '</div>',
            unsafe_allow_html=True,
        )


with _info_right.expander("ℹ️ Qu'est-ce qu'AI Energy Score ?", expanded=False):
    st.markdown(f"""
    **AI Energy Score** est un classement publié par **Hugging Face** qui **mesure** l'énergie réellement consommée
    par des modèles d'IA, dans les mêmes conditions pour tous :
    - un même processeur graphique (**GPU NVIDIA H100**), une requête à la fois ;
    - **1 000 requêtes** par tâche (génération de texte, raisonnement…) ;
    - seule l'énergie du GPU est comptée, exprimée en **Wh par requête** ;
    - une note de **1 à 5 étoiles**, relative à la classe de taille du modèle
      (A : moins de 20 milliards de paramètres, B : de 20 à 66, C : plus de 66).

    Seuls les modèles **open-weight** (dont les poids sont publics) peuvent être mesurés : GPT, Claude ou Gemini n'y figurent pas.
    Ce dashboard s'en sert pour **vérifier les estimations de ComparIA** sur les {len(comparable)} modèles présents
    dans les deux sources (section « 🔬 Estimé vs mesuré »).
    """)

# ══════════════════════════════════════════════════════════════════════════════
# VUE D'ENSEMBLE — 4 KPI tiles
# ══════════════════════════════════════════════════════════════════════════════
st.header("Vue d'ensemble")

c1, c2, c3, c4 = st.columns(4)

_delta = nb_models_energy - nb_models_score

_CARD = "border:1px solid #e8e8e8;border-radius:10px;padding:16px;text-align:center;background:#fafafa;height:150px;display:flex;flex-direction:column;justify-content:center;align-items:center;"

c1.markdown(f"""
    <div style="{_CARD}">
        <div style="font-size:11px;color:#888;margin-bottom:8px;">💬 CONVERSATIONS EN BASE</div>
        <div style="font-size:50px;font-weight:700;margin:6px 0;">{health["nb_conversations"]:,}</div>
    </div>
""", unsafe_allow_html=True)

c2.markdown(f"""
    <div style="{_CARD}">
        <div style="font-size:11px;color:#888;margin-bottom:8px;">🤖 MODÈLES AU CLASSEMENT</div>
        <div style="font-size:50px;font-weight:700;margin:6px 0;">{nb_models_score}</div>
    </div>
""", unsafe_allow_html=True)

c3.markdown(f"""
    <div style="{_CARD}">
        <div style="font-size:11px;color:#888;margin-bottom:8px;">⚡ MODÈLES AVEC DONNÉES ÉNERGIE</div>
        <div style="font-size:50px;font-weight:700;margin:0px 0;">{nb_models_energy}</div>
        <div style="font-size:12px;color:#E63946;">{_delta} vs modèles totaux</div>
    </div>
""", unsafe_allow_html=True)

top_prov = top_model["provider"]
with c4:
    st.markdown(
        f"""
        <div style="border:1px solid #e8e8e8;border-radius:10px;padding:16px;text-align:center;background:#fafafa;height:150px;display:flex;flex-direction:column;justify-content:center;align-items:center;">
            <div style="font-size:11px;color:#888;margin-bottom:8px;">🏆 TOP MODÈLE</div>
            {logo_img_tag(top_prov, size=36)}
            <div style="font-size:24px;font-weight:700;margin:6px 0;">{top_model['score']:.1f}%</div>
            <div style="font-size:11px;color:#666;">{top_model['model']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SCORE AJUSTÉ
# ══════════════════════════════════════════════════════════════════════════════
st.header("🏅 Performance des modèles")

with st.expander("📐 Méthodologie — Score ajusté", expanded=False):
    st.markdown("""
    Chaque duel oppose deux modèles sur une même conversation. Plutôt qu'un simple win rate,
    on attribue **1 pt** pour une victoire, **0.5 pt** pour une égalité et **0 pt** pour une défaite —
    ce qui intègre les ~32 % de duels se terminant à égalité. Le score est la moyenne pondérée sur
    l'ensemble des duels (min. 200 duels), avec un **intervalle de confiance à 95 %**.
    """)

_col_slider, _, _, _ = st.columns([1, 1, 1, 1])
with _col_slider:
    n_top = st.slider("Nombre de modèles", min_value=5, max_value=30, value=15, step=1, key="score_slider")

top10 = classement.head(n_top).sort_values("score", ascending=True).reset_index(drop=True)
y_pos = list(range(len(top10)))
bar_colors = [PROVIDER_COLORS.get(p, "#AAAAAA") for p in top10["provider"]]

fig_score = go.Figure()
fig_score.add_trace(go.Bar(
    x=top10["score"],
    y=y_pos,
    orientation="h",
    marker_color=bar_colors,
    showlegend=False,
    error_x=dict(type="data", array=top10["ic"].tolist(), visible=True, color="#555", thickness=1.5, width=4),
    customdata=top10[["ic", "duels", "provider"]].values,
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Score : %{x:.1f}% ± %{customdata[0]:.1f}%<br>"
        "Duels : %{customdata[1]:,}<br>"
        "Provider : %{customdata[2]}<extra></extra>"
    ),
    text=top10["model"].tolist(),
    textposition="none",
))

# Score annotations à droite des barres
for i, (_, row) in enumerate(top10.iterrows()):
    fig_score.add_annotation(
        x=row["score"] + row["ic"] + 0.5, y=i,
        text=f"{row['score']:.1f}%",
        xanchor="left", yanchor="middle",
        showarrow=False, font=dict(size=11),
        xref="x", yref="y",
    )

# Logos providers à gauche des barres
_LOGO_X = -6
for i, (_, row) in enumerate(top10.iterrows()):
    prov = row["provider"]
    if prov in logo_b64:
        fig_score.add_layout_image(
            source=logo_b64[prov],
            xref="x", yref="y",
            x=_LOGO_X, y=i,
            sizex=4, sizey=0.65,
            xanchor="center", yanchor="middle",
            layer="above",
        )

# Légende providers
for prov in top10["provider"].unique():
    fig_score.add_trace(go.Bar(
        x=[None], y=[None], orientation="h",
        name=prov,
        marker_color=PROVIDER_COLORS.get(prov, "#AAAAAA"),
        showlegend=True,
    ))

x_max = top10["score"].max() + top10["ic"].max() + 10
fig_score.update_layout(
    title=f"Top {n_top} modèles — Score ajusté avec intervalle de confiance à 95%<br>"
          "<sup>victoire=1pt, égalité=0.5pt | min. 200 duels</sup>",
    xaxis=dict(
        title="Score ajusté (%)",
        range=[-10, x_max],
        tickvals=[t for t in range(0, int(x_max) + 5, 10) if t >= 0],
    ),
    yaxis=dict(
        tickvals=y_pos,
        ticktext=top10["model"].tolist(),
        tickfont=dict(size=11),
    ),
    height=max(350, n_top * 38),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(title="Provider", orientation="v"),
    barmode="overlay",
    margin=dict(l=180),
)
st.plotly_chart(fig_score, use_container_width=True)

st.markdown("""
**Key insights**
- 🥇 **gemini-2.0-flash** domine avec un score de ~65 % et un intervalle de confiance étroit grâce à plus de 6 000 duels
- 🔵 **Google** est le provider le plus représenté dans le top 10 (6 modèles), devant **DeepSeek** et **Mistral** (2 chacun)
- 🔧 La correction de 9 182 votes au champ `both_equal` vide fait remonter des modèles de fin 2024 (*gemma-2-27b-it-q8* 3e, *gemini-1.5-pro* 4e)
- ⚖️ Du 11e au 30e rang, les scores sont très proches (54–58 %) avec des intervalles d'environ ±2,5 pts : leur ordre exact est peu significatif
""")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONSOMMATION ÉNERGÉTIQUE
# ══════════════════════════════════════════════════════════════════════════════
st.header("⚡ Consommation énergétique des modèles")

with st.expander("📐 Méthodologie — Consommation énergétique", expanded=False):
    st.markdown("""
    La colonne `total_conv_X_kwh` est une **estimation** de ComparIA, pas une mesure : elle vaut le nombre de tokens
    générés × un coefficient fixe propre à chaque modèle. On calcule la moyenne par modèle (réponses vides exclues)
    et on la divise par le nombre moyen de tours (`avg_turns`) pour obtenir une **consommation par requête**
    comparable à d'autres usages numériques. La consommation par token (somme des kWh ÷ somme des tokens)
    est donc le coefficient de ComparIA lui-même.
    """)

CRITERIA = {
    "💬 Par conversation":  ("wh_conv",          "Wh / conversation"),
    "📨 Par requête":       ("wh_req",           "Wh / requête"),
    "🔬 Par token généré":  ("mwh_per_token",    "mWh / token"),
}

_ctrl_left, _ctrl_right = st.columns([1, 1])
with _ctrl_left:
    n_energy = st.slider("Nombre de modèles", min_value=5, max_value=len(energy_g), value=15, step=1, key="energy_slider")
with _ctrl_right:
    selected_criterion = st.selectbox(
        "Critère de consommation",
        options=list(CRITERIA.keys()),
        index=0,
    )
col_name, col_label = CRITERIA[selected_criterion]

# Données triées pour ce critère
energy_sorted = (
    energy_g[["model", "provider", col_name]]
    .dropna(subset=[col_name])
    .sort_values(col_name, ascending=True)
    .head(n_energy)
    .reset_index(drop=True)
)

col_left, col_right = st.columns([1, 9])

# ── Top 3 ranking (colonne gauche) ────────────────────────────────────────────
with col_left:
    st.markdown("**Top 3**<br><small>plus sobres</small>", unsafe_allow_html=True)
    medals = ["🥇", "🥈", "🥉"]
    for rank, (_, row) in enumerate(energy_sorted.head(3).iterrows()):
        val = row[col_name]
        val_str = fmt_value(val)
        st.markdown(
            f"""
            <div style="padding:10px 4px;border-bottom:1px solid #f0f0f0;text-align:center;">
                <div style="font-size:22px;">{medals[rank]}</div>
                {logo_img_tag(row['provider'], size=28)}
                <div style="font-size:9px;color:#555;margin-top:4px;word-break:break-all;line-height:1.3;">
                    {row['model']}
                </div>
                <div style="font-size:11px;font-weight:700;color:{PROVIDER_COLORS.get(row['provider'],'#444')};margin-top:2px;">
                    {val_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Bar chart (colonne droite) ────────────────────────────────────────────────
with col_right:
    bar_colors_e = [PROVIDER_COLORS.get(p, "#AAAAAA") for p in energy_sorted["provider"]]
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Bar(
        x=energy_sorted[col_name],
        y=energy_sorted["model"],
        orientation="h",
        marker_color=bar_colors_e,
        customdata=energy_sorted[["provider", col_name]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{col_label} : %{{x:.3f}}<br>"
            "Provider : %{customdata[0]}<extra></extra>"
        ),
        text=energy_sorted[col_name].map(fmt_value),
        textposition="outside",
    ))
    fig_energy.update_layout(
        title=f"Consommation des modèles — {col_label}",
        xaxis_title=col_label,
        height=max(500, len(energy_sorted) * 22),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10),
    )
    st.plotly_chart(fig_energy, use_container_width=True)

st.markdown("""
**Key insights**
- 💡 Les plus sobres **par token** (~0,08 mWh) sont de petits modèles ou des modèles à experts : *lfm2*, *gpt-oss-20b*, *qwen3-30b-a3b*, *gpt-5-nano*, *gemma-3-4b*
- 📊 **Par conversation**, l'écart est énorme : de 0,03 Wh (*chocolatine*) à 150 Wh (*grok-4.20*), pour une médiane de 0,9 Wh
- ⚠️ Les modèles de raisonnement consomment beaucoup plus par conversation (*deepseek-r1* 5×, *kimi-k2-thinking* 8× la médiane) pour des scores proches de la moyenne (54 % et 51 %, moyenne 49 %)
- 🔎 Ces valeurs sont des **estimations** de ComparIA (tokens × coefficient par modèle) : le classement par token reflète ce coefficient, pas une mesure
""")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PARETO : QUALITÉ VS EFFICIENCE
# ══════════════════════════════════════════════════════════════════════════════
st.header("🎯 Synthèse : Performance vs consommation énergétique")
st.markdown(
    "La **frontière de Pareto** relie les modèles qu'aucun autre ne domine simultanément sur les deux axes. "
    "Un modèle est sur cette frontière si améliorer son score impliquerait nécessairement d'accepter une consommation "
    "plus élevée — et inversement. C'est parmi ces modèles non dominés que se trouvent les meilleurs compromis "
    "qualité / efficience énergétique."
)

# Champions
champion_qualite = merged.loc[merged["score"].idxmax()]

pareto = pareto_frontier(merged)

# Champion efficience : point de coude (knee) de la frontière de Pareto
# = modèle qui maximise (score_normalisé - kwh_normalisé)
# → meilleur compromis entre qualité et sobriété, indépendamment des seuils arbitraires
_s_min, _s_max = pareto["score"].min(), pareto["score"].max()
_k_min, _k_max = pareto["avg_kwh"].min(), pareto["avg_kwh"].max()
_pareto_knee = pareto.copy()
_pareto_knee["_knee"] = (
    (_pareto_knee["score"]   - _s_min) / (_s_max - _s_min + 1e-12)
    - (_pareto_knee["avg_kwh"] - _k_min) / (_k_max - _k_min + 1e-12)
)
champion_efficience = _pareto_knee.loc[_pareto_knee["_knee"].idxmax()]

# 3e champion : le plus sobre PAR TOKEN parmi les modèles de bonne qualité.
# Remplace l'ancien « meilleur ratio score / kWh », qui désignait un modèle très sobre mais faible (29 % de score).
# Seuil de 55 % : nettement au-dessus de la moyenne des modèles (~49 %). Le critère par token ne dépend pas
# de la longueur des réponses : il compare l'efficience des modèles eux-mêmes (comme dans le notebook d'origine).
QUALITY_THRESHOLD = 55
good_models = merged[merged["score"] >= QUALITY_THRESHOLD]
champion_sobre = good_models.loc[good_models["mwh_per_token"].idxmin()]

col_chart, col_rank = st.columns([6, 4])

with col_chart:
    fig_pareto = go.Figure()

    # Tous les modèles (gris)
    fig_pareto.add_trace(go.Scatter(
        x=merged["score"], y=merged["wh_conv"],
        mode="markers",
        marker=dict(color="#CCCCCC", size=7),
        text=merged["model"],
        hovertemplate="<b>%{text}</b><br>Score : %{x:.1f}%<br>Wh/conversation : %{y:.3f}<extra></extra>",
        name="Autres modèles",
        showlegend=True,
    ))

    # Frontière de Pareto
    fig_pareto.add_trace(go.Scatter(
        x=pareto["score"], y=pareto["wh_conv"],
        mode="lines+markers",
        line=dict(color="#333", dash="dash", width=1.5),
        marker=dict(color="#333", size=7),
        name="Frontière de Pareto",
        hovertemplate="<b>%{text}</b><br>Score : %{x:.1f}%<br>Wh/conversation : %{y:.3f}<extra></extra>",
        text=pareto["model"],
    ))

    # 3 champions
    for key, champ, label in [
        ("qualite",    champion_qualite,    "Meilleure qualité"),
        ("efficience", champion_efficience, "Meilleur rapport qualité/énergie"),
        ("token",      champion_sobre,      f"Le plus sobre par token (score ≥ {QUALITY_THRESHOLD} %)"),
    ]:
        color = CHAMPION_COLORS[key]
        fig_pareto.add_trace(go.Scatter(
            x=[champ["score"]], y=[champ["wh_conv"]],
            mode="markers+text",
            marker=dict(color=color, size=14, line=dict(color="white", width=2)),
            text=[champ["model"].split("-")[0]],
            textposition="top center",
            textfont=dict(color=color, size=10),
            name=label,
            customdata=[[champ["model"], label]],
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>{label}<br>Score : %{{x:.1f}}%<br>Wh/conversation : %{{y:.3f}}<extra></extra>",
        ))

    # Ligne verticale : gemini-2.0-flash (meilleur score), affichée seulement s'il n'a pas de données énergie.
    # Sur l'échantillon du notebook il n'en avait pas ; sur les données complètes il en a, et il figure dans le graphique.
    _g2_score = classement.loc[classement["model"] == "gemini-2.0-flash", "score"]
    if not _g2_score.empty and "gemini-2.0-flash" not in set(merged["model"]):
        _g2_score_val = float(_g2_score.iloc[0])
        fig_pareto.add_vline(
            x=_g2_score_val,
            line=dict(color="#AAAAAA", dash="dot", width=1.5),
        )
        fig_pareto.add_annotation(
            x=_g2_score_val,
            y=1,
            yref="paper",
            text="gemini-2.0-flash<br><i>données énergie non disponibles</i>",
            showarrow=False,
            font=dict(size=9, color="#888"),
            xanchor="right",
            yanchor="top",
            bgcolor="white",
            bordercolor="#CCCCCC",
            borderwidth=1,
            borderpad=3,
        )

    # Seuil de « bonne qualité » utilisé par le 3e champion : ligne verticale pour situer la zone concernée
    fig_pareto.add_vline(
        x=QUALITY_THRESHOLD,
        line=dict(color="#E9C46A", dash="dot", width=1.5),
        annotation_text=f"seuil de bonne qualité ({QUALITY_THRESHOLD} %)",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color="#B8962E"),
    )

    _x_max = max(merged["score"].max(), float(_g2_score.iloc[0]) if not _g2_score.empty else 0) + 2

    fig_pareto.update_layout(
        title="Performance vs consommation énergétique<br><sup>La frontière de Pareto identifie les modèles non dominés</sup>",
        xaxis=dict(title="Score ajusté (%)", range=[merged["score"].min() - 2, _x_max]),
        yaxis_title="Wh / conversation (échelle log)",
        yaxis_type="log",
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

with col_rank:
    st.markdown("#### Modèles champions")
    _n_energy    = len(merged)
    _eff_s_norm  = float((champion_efficience["score"]   - _s_min) / (_s_max - _s_min + 1e-12))
    _eff_k_norm  = float((champion_efficience["avg_kwh"] - _k_min) / (_k_max - _k_min + 1e-12))

    champions_data = [
        (
            champion_qualite, "qualite", "🏆 Meilleure qualité brute",
            f"Score ajusté = {champion_qualite['score']:.1f}% — le plus élevé parmi les {_n_energy} modèles avec données énergétiques.",
            "Parmi les modèles avec données de consommation disponibles",
        ),
        (
            champion_efficience, "efficience", "⚡ Meilleur compromis qualité / énergie",
            f"Score normalisé ({_eff_s_norm:.2f}) − énergie normalisée ({_eff_k_norm:.2f}) = {_eff_s_norm - _eff_k_norm:.2f} — maximum sur la frontière de Pareto.",
            None,
        ),
        (
            champion_sobre, "token", f"🔬 Le plus sobre par token (score ≥ {QUALITY_THRESHOLD} %)",
            f"{fmt_value(champion_sobre['mwh_per_token'])} mWh par token : la plus faible consommation par token parmi les "
            f"{len(good_models)} modèles ayant au moins {QUALITY_THRESHOLD} % de score.",
            None,
        ),
    ]

    for champ, key, label, reason, caveat in champions_data:
        color = CHAMPION_COLORS[key]
        prov  = champ["provider"]
        st.markdown(
            f'<div style="border-left:5px solid {color};background:#fafafa;border-radius:8px;padding:14px 16px;margin-bottom:14px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
            f'<div style="font-size:11px;font-weight:600;color:{color};">{label}</div>'
            + (f'<div style="font-size:9px;color:#aaa;font-style:italic;">⚠️ {caveat}</div>' if caveat else '')
            + '</div>'
            f'<div style="display:flex;gap:12px;align-items:flex-start;">'
            f'<div style="flex:0 0 44%;">'
            f'{logo_img_tag(prov, size=30)}'
            f'<div style="font-weight:700;font-size:12px;margin-top:5px;">{champ["model"]}</div>'
            f'<div style="font-size:11px;color:#666;">{prov}</div>'
            f'</div>'
            f'<div style="flex:1;border-left:1px solid #eee;padding-left:12px;">'
            f'<div style="display:flex;gap:12px;margin-bottom:7px;">'
            f'<div><div style="font-size:9px;color:#aaa;">SCORE</div>'
            f'<div style="font-weight:700;font-size:15px;color:{color};">{champ["score"]:.1f}%</div></div>'
            f'<div><div style="font-size:9px;color:#aaa;">Wh/CONV</div>'
            f'<div style="font-weight:700;font-size:15px;color:{color};">{fmt_value(champ["wh_conv"])}</div></div>'
            f'</div>'
            f'<div style="font-size:10px;color:#666;line-height:1.5;">{reason}</div>'
            + '</div></div></div>',
            unsafe_allow_html=True,
        )

with st.expander("📐 Méthodologie — Synthèse Pareto", expanded=False):
    st.markdown("""
    Le graphique croise **score ajusté** (axe X) et **consommation énergétique par conversation** (axe Y, échelle log).
    La **frontière de Pareto** relie les modèles qu'aucun autre ne domine simultanément sur les deux axes :
    un modèle est sur cette frontière si améliorer son score impliquerait nécessairement d'accepter une consommation plus élevée.

    Les **3 champions** sont sélectionnés selon des critères distincts :
    - 🏆 **Meilleure qualité** — score ajusté le plus élevé parmi les modèles avec données énergétiques
    - ⚡ **Meilleur compromis** — point de coude de la frontière de Pareto : score normalisé − kWh normalisé maximal
    - 🔬 **Le plus sobre par token** — consommation par token la plus faible parmi les modèles ayant au moins 55 % de score.
      Il remplace l'ancien « meilleur ratio score / kWh », qui désignait un modèle très sobre mais peu performant (29 % de score).

    **Pourquoi un seuil de 55 % ?** Le score moyen des modèles est d'environ 49,6 %. Un modèle à 55 % dépasse cette moyenne
    de plus de 5 points, soit bien plus que l'incertitude de son score (intervalle de confiance médian de ±1,8 point pour
    ces modèles) : même dans le pire cas, il reste au-dessus de la moyenne. On ne retient donc que des modèles
    **nettement meilleurs que la moyenne**, sans être trop exigeant : 21 modèles sur 110 passent ce seuil (ligne jaune verticale).

    **Pourquoi ce champion n'est-il pas sur la frontière de Pareto ?** La frontière est calculée sur les deux axes du graphique :
    le score et l'énergie **par conversation**. Le 3e champion est choisi sur un autre critère, l'énergie **par token**.
    Par conversation, *gemma-3-12b* est dépassé par *gemma-2-27b-it-q8* (meilleur score et conversations moins coûteuses) ;
    par token, il est en revanche sur la frontière (aucun modèle ne fait mieux sur les deux critères à la fois).
    Les deux critères ne disent pas la même chose : par conversation, on mesure le coût réel d'un échange,
    qui dépend aussi de la longueur des réponses ; par token, on mesure l'efficience du modèle lui-même.
    """)

st.markdown("""
**Key insights**
- 📐 La **frontière de Pareto** montre qu'aucun modèle unique ne domine sur tous les critères — le choix optimal dépend des priorités (qualité vs coût)
- ⚡ Le meilleur compromis (*gemma-2-27b-it-q8* : 62 % pour 0,07 Wh par conversation) approche le score du n°1 (*gemini-2.0-flash*, 64,7 %) pour près de 90 fois moins d'énergie
- 🔬 Parmi les modèles de bonne qualité (≥ 55 %), *gemma-3-12b* est le plus sobre par token : un bon choix pour un usage à fort volume
""")

with st.expander("⚠️ Limites de cette analyse"):
    st.markdown("""
    - **Énergie estimée, pas mesurée** : ComparIA calcule kWh = tokens × coefficient par modèle ; la comparaison avec des mesures réelles (AI Energy Score) n'est possible que pour 11 modèles open-weight
    - **Données énergétiques presque complètes** : 112 modèles ont des données de consommation ; 4 modèles de 2024 n'en ont aucune (*c4ai-command-r-08-2024*, *ministral-8b-instruct-2410*, *mistral-large-2411*, *qwen2.5-coder-32b-instruct*)
    - **Biais de sélection** : seuls les modèles avec ≥ 200 duels sont inclus dans le classement — les modèles récents peuvent être sous-représentés
    - **Qualité subjective** : le score est basé sur des préférences humaines, qui peuvent varier selon la langue, la culture et le contexte des conversations
    - **Snapshot temporel** : les modèles évoluent continuellement — ce classement reflète leurs performances à la date de collecte des données
    - **Consommation estimée par token** : le `mwh_per_token` est calculé globalement par modèle, sans distinction selon le type de tâche ou la longueur des échanges
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 BIS — ESTIMÉ VS MESURÉ (ComparIA × AI Energy Score)
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔬 Estimé vs mesuré : peut-on se fier aux estimations ?")
st.markdown(
    "ComparIA **estime** l'énergie de chaque modèle ; AI Energy Score la **mesure** sur un vrai GPU. "
    "Si les estimations sont fiables, les deux sources doivent **classer les modèles dans le même ordre**."
)


def spearman(x, y):
    """Corrélation de Spearman = corrélation de Pearson calculée sur les classements (rangs)."""
    return x.rank().corr(y.rank())


exact = comparable[comparable["match_type"] == "exact"]
rho_all = spearman(comparable["comparia_wh_per_req"], comparable["gpu_energy_wh"])
rho_exact = spearman(exact["comparia_wh_per_req"], exact["gpu_energy_wh"])

MATCH_COLORS = {"exact": "#2A9D8F", "nom_different": "#E9C46A", "version_base": "#E76F51"}
MATCH_LABELS = {
    "exact": "Nom identique",
    "nom_different": "Même modèle, nom différent",
    "version_base": "Mesure sur la version de base",
}

col_scatter, col_rho = st.columns([7, 3])

with col_scatter:
    fig_evm = go.Figure()
    for match_type, group in comparable.groupby("match_type"):
        fig_evm.add_trace(go.Scatter(
            x=group["gpu_energy_wh"], y=group["comparia_wh_per_req"],
            mode="markers+text",
            text=group["model"], textposition="top center", textfont=dict(size=9),
            marker=dict(size=12, color=MATCH_COLORS.get(match_type, "#AAAAAA"), line=dict(color="white", width=1)),
            name=MATCH_LABELS.get(match_type, match_type),
            hovertemplate="<b>%{text}</b><br>Mesuré : %{x:.3f} Wh<br>Estimé : %{y:.3f} Wh<extra></extra>",
        ))
    fig_evm.update_layout(
        title=f"Énergie estimée vs mesurée — génération de texte ({len(comparable)} modèles)"
              "<br><sup>échelles logarithmiques · Wh par requête</sup>",
        xaxis=dict(title="Énergie MESURÉE — AI Energy Score (Wh / requête, GPU H100)", type="log"),
        yaxis=dict(title="Énergie ESTIMÉE — ComparIA (Wh / requête)", type="log"),
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    st.plotly_chart(fig_evm, use_container_width=True)

with col_rho:
    st.metric("Corrélation de Spearman", f"{rho_all:.2f}")
    st.caption(f"{len(comparable)} modèles présents dans les deux sources")
    st.metric("Sur les correspondances exactes", f"{rho_exact:.2f}")
    st.caption(f"{len(exact)} modèles portant le même nom dans les deux sources")
    st.markdown(
        "**Lecture** : **+1** = les deux sources classent les modèles exactement dans le même ordre ; "
        "**0** = aucun lien entre les deux classements."
    )

# Tableau détaillé : un modèle par ligne, trié du plus sobre au plus gourmand SELON LA MESURE.
# Si l'estimation est fiable, la colonne « Estimé » doit elle aussi grandir de haut en bas.
LINK_LABELS = {
    "exact": "✅ Même modèle, même nom",
    "nom_different": "✅ Même modèle, nom différent",
    "version_base": "⚠️ Même modèle, autre version",
}


def gap_cell(ratio):
    """Écart estimé ÷ mesuré. On utilise un rapport plutôt qu'une différence : les valeurs vont
    de 0,02 à 1,3 Wh, et une différence de 0,03 Wh serait énorme pour un petit modèle mais
    négligeable pour un gros. Vert : moins d'un facteur 3 d'écart ; rouge : plus d'un facteur 3."""
    text = f"× {ratio:.1f} (surestimé)" if ratio >= 1 else f"÷ {1 / ratio:.1f} (sous-estimé)"
    color, background = ("#1B7F4B", "#E6F4EA") if 1 / 3 <= ratio <= 3 else ("#C0392B", "#FDECEA")
    return f'<td style="padding:6px;color:{color};background:{background};font-weight:600;">{text}</td>'


_th = 'style="padding:6px;text-align:left;border-bottom:2px solid #ddd;font-size:12px;"'
_rows = ""
for _, row in comparable.sort_values("gpu_energy_wh").iterrows():
    ratio = row["comparia_wh_per_req"] / row["gpu_energy_wh"]
    _rows += (
        '<tr style="border-bottom:1px solid #f0f0f0;">'
        f'<td style="padding:6px;">{logo_img_tag(row["provider"], size=22)} '
        f'<span style="font-size:11px;color:#666;">{row["provider"]}</span></td>'
        f'<td style="padding:6px;font-weight:600;">{row["model"]}</td>'
        f'<td style="padding:6px;font-size:12px;">{LINK_LABELS.get(row["match_type"], row["match_type"])}</td>'
        f'<td style="padding:6px;">{row["gpu_energy_wh"]:.3f}</td>'
        f'<td style="padding:6px;">{row["comparia_wh_per_req"]:.3f}</td>'
        f'{gap_cell(ratio)}'
        '</tr>'
    )
st.markdown(
    '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
    f'<tr><th {_th}>Fournisseur</th><th {_th}>Modèle</th><th {_th}>Lien entre les sources</th>'
    f'<th {_th}>Mesuré — AI Energy Score<br>(Wh / requête)</th>'
    f'<th {_th}>Estimé — ComparIA<br>(Wh / requête)</th>'
    f'<th {_th}>Écart estimé / mesuré</th></tr>'
    f'{_rows}</table>',
    unsafe_allow_html=True,
)
st.caption(
    "Lecture : le tableau est trié par la mesure ; si l'estimation est fiable, la colonne « Estimé » grandit aussi. "
    "« Lien entre les sources » indique la qualité de l'appariement : ⚠️ = même modèle, mais pas la même version "
    "dans les deux sources (voir la méthodologie) ; la comparaison est alors approximative. "
    "« Écart » : combien de fois l'estimation est au-dessus (×) ou en dessous (÷) de la mesure ; "
    "vert = moins d'un facteur 3, rouge = plus d'un facteur 3."
)

with st.expander("📐 Méthodologie — Estimé vs mesuré", expanded=False):
    st.markdown("""
    - Les modèles des deux sources sont reliés par une **table de correspondance** écrite à la main
      (`references/model_mapping.csv`), car leurs noms diffèrent (`gemma-3-27b` ↔ `google/gemma-3-27b-pt`).
    - On compare **à tâche égale** (génération de texte) et dans la **même unité** (Wh par requête) :
      l'estimation de ComparIA en kWh par requête est convertie en Wh.
    - Les deux sources n'ont ni les mêmes conditions (vraies conversations contre 1 000 requêtes de test sur un GPU H100)
      ni les mêmes réglages : seul **l'ordre** des modèles est comparable. D'où la **corrélation de Spearman**,
      une corrélation de Pearson calculée sur les classements plutôt que sur les valeurs.
    - **« ⚠️ Même modèle, autre version »** (5 modèles) : un modèle existe souvent en deux versions. La **version de base**
      est le modèle tel qu'il sort de son premier entraînement (il sait compléter un texte) ; la **version conversationnelle**
      (« instruct ») est ce même modèle, réentraîné pour dialoguer. Les deux ont la même taille et la même architecture,
      donc le même coût par token : ce n'est pas un autre modèle. Mais AI Energy Score a mesuré la version de base, alors que
      ComparIA utilise la version conversationnelle, qui ne répond pas forcément avec la même longueur. D'où le ⚠️ :
      la comparaison est approximative. Le calcul refait sur les seules correspondances « même nom » (ρ = 1,00)
      vérifie que cette approximation ne change pas la conclusion.
    - **Colonne « Écart »** : on compare l'estimation à la mesure par un **rapport** (« × 1,9 » = l'estimation est
      1,9 fois plus élevée que la mesure ; « ÷ 7,5 » = 7,5 fois plus basse), et non par une différence. Les valeurs vont
      de 0,02 à 1,3 Wh : une différence de 0,03 Wh serait énorme pour un petit modèle et négligeable pour un gros,
      alors qu'un rapport se lit de la même façon pour tous. **Vert** : moins d'un facteur 3 d'écart, l'estimation est
      du bon ordre de grandeur ; **rouge** : plus d'un facteur 3, l'estimation s'écarte nettement de la mesure.
    """)

st.markdown(f"""
**Key insights**
- ✅ Avec ρ = {rho_all:.2f}, ComparIA classe les modèles presque dans le même ordre que les mesures réelles : ses estimations sont **fiables pour comparer** des modèles entre eux, pas pour connaître leur consommation exacte
- 🎯 Sur les seules correspondances exactes, ρ = {rho_exact:.2f} : l'approximation « version de base / instruct » ne change pas la conclusion
- 🔎 Deux points s'écartent : *mixtral-8x7b* (mesuré sur sa version de base, sans doute plus bavarde) et *deepseek-r1-distill-llama-70b* (modèle de raisonnement, très verbeux en conversation réelle)
- ⚠️ Seuls {len(comparable)} modèles, tous open-weight, peuvent être comparés : les modèles propriétaires ne sont pas mesurables
""")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANALYSE PAR CATÉGORIE
# ══════════════════════════════════════════════════════════════════════════════
col_cat, col_energy = st.columns([1, 1])

GOOGLE_KWH = 0.0003
IPHONE_KWH = 0.022
LED_KWH    = 0.009

def fmt_led(kwh):
    s = kwh / LED_KWH * 3600
    if s < 60:     return f"{s:.0f} secondes"
    elif s < 3600: return f"{s/60:.1f} minutes"
    else:          return f"{s/3600:.2f} heures"

def fmt_google(kwh):
    g = kwh / GOOGLE_KWH
    return f"{g*100:.1f}% d'une recherche" if g < 1 else f"{g:.2f} recherches"

def fmt_iphone(kwh):
    return f"{kwh / IPHONE_KWH * 100:.2f}% d'une recharge"

CRITERIA_CAT = {
    "🔬 Par token généré":  ("mwh_per_token",   "mWh / token"),
    "💬 Par conversation":  ("wh_conv",          "Wh / conversation"),
    "📨 Par requête":       ("wh_req",           "Wh / requête"),
}

with col_cat:
    st.header("🗂️ Combien coûte une requête selon la catégorie ?")
    categories = sorted(ranking_cat["categories"].unique())
    _fc1, _fc2 = st.columns([1, 1])
    with _fc1:
        selected_cat = st.selectbox("Catégorie", options=categories, index=0, key="cat_sel")
    with _fc2:
        selected_cat_crit = st.selectbox("Critère", options=list(CRITERIA_CAT.keys()), index=0, key="cat_crit")
    cat_col, cat_label = CRITERIA_CAT[selected_cat_crit]

    cat_data = (
        ranking_cat[ranking_cat["categories"] == selected_cat]
        .dropna(subset=[cat_col])
        .sort_values(cat_col, ascending=True)
        .head(5)
        .reset_index(drop=True)
    )

    y_cat = list(range(len(cat_data)))
    bar_colors_cat = [PROVIDER_COLORS.get(p, "#AAAAAA") for p in cat_data["provider"]]

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        x=cat_data[cat_col],
        y=y_cat,
        orientation="h",
        marker_color=bar_colors_cat,
        customdata=cat_data[["provider", "nb_conv_cat"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"{cat_label} : %{{x:.3f}}<br>"
            "Provider : %{customdata[0]}<br>"
            "Conversations : %{customdata[1]:,}<extra></extra>"
        ),
        text=cat_data["model"].tolist(),
        textposition="none",
        showlegend=False,
    ))

    # Annotations valeurs
    for i, (_, row) in enumerate(cat_data.iterrows()):
        val = row[cat_col]
        fig_cat.add_annotation(
            x=val * 1.02, y=i,
            text=fmt_value(val),
            xanchor="left", yanchor="middle",
            showarrow=False, font=dict(size=9),
            xref="x", yref="y",
        )

    # Logos providers
    _x_max_cat = cat_data[cat_col].max()
    _logo_x_cat = -_x_max_cat * 0.18
    for i, (_, row) in enumerate(cat_data.iterrows()):
        prov = row["provider"]
        if prov in logo_b64:
            fig_cat.add_layout_image(
                source=logo_b64[prov],
                xref="x", yref="y",
                x=_logo_x_cat, y=i,
                sizex=abs(_logo_x_cat) * 0.9,
                sizey=0.65,
                xanchor="center", yanchor="middle",
                layer="above",
            )

    fig_cat.update_layout(
        title=f"Top 5 modèles les plus économes ({cat_label})<br><sup>{selected_cat} — min. 30 conversations</sup>",
        xaxis=dict(
            title=cat_label,
            range=[_logo_x_cat * 1.6, _x_max_cat * 1.35],
            tickvals=[round(v, 5) for v in np.linspace(0, _x_max_cat, 5)],
        ),
        yaxis=dict(
            tickvals=y_cat,
            ticktext=cat_data["model"].tolist(),
            tickfont=dict(size=9),
            autorange="reversed",
        ),
        height=340,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=160),
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    with st.expander("📐 Méthodologie — Analyse par catégorie", expanded=False):
        st.markdown("""
        Les conversations de ComparIA sont classées en catégories thématiques (sciences, droit, arts…).
        Pour chaque catégorie, on retient uniquement les modèles ayant participé à **au moins 30 conversations**
        afin d'assurer une représentativité statistique minimale.

        La métrique par défaut est le **mWh par token généré** (`mwh_per_token`), calculée globalement par modèle
        (endpoint `/categories` de l'API). C'est le coefficient fixé par ComparIA pour chaque modèle : il est
        **identique dans toutes les catégories**. Le top 5 montre donc quels modèles sobres sont présents
        dans la catégorie sélectionnée, pas une consommation propre à cette catégorie.
        """)
    st.markdown("""
**Key insights**
- 🔬 Les mêmes modèles sobres reviennent partout : *lfm2-8b-a1b*, *gpt-oss-20b*, *qwen3-30b-a3b* et *phi-3.5-mini-instruct* sont dans le top 5 de 18 catégories sur 20
- 📂 Le mWh/token d'un modèle ne change pas d'une catégorie à l'autre (coefficient ComparIA) : ce qui change, c'est quels modèles sont présents
- ⚖️ Un modèle économe en tokens sur une catégorie peut être moins performant sur le score ajusté — l'efficience ne garantit pas la qualité
""")

with col_energy:
    st.header("⚡ Combien coûte une requête pour ce modèle ?")
    energy_models = sorted(energy_g["model"].tolist())
    selected_model = st.selectbox("Modèle", options=energy_models, key="energy_model_sel")
    kwh = float(energy_g.loc[energy_g["model"] == selected_model, "avg_kwh_per_req"].fillna(
        energy_g.loc[energy_g["model"] == selected_model, "avg_kwh"]
    ).iloc[0])

    _card_e = "border-radius:10px;padding:14px;text-align:center;margin-bottom:10px;"
    st.markdown(
        f'<div style="{_card_e}background:#fff8f0;border:1px solid #F4A261;">'
        f'<div style="font-size:28px;">💡</div>'
        f'<div style="font-size:11px;color:#888;margin:4px 0;">AMPOULE LED 60W</div>'
        f'<div style="font-size:18px;font-weight:700;color:#F4A261;">{fmt_led(kwh)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="{_card_e}background:#f0f4ff;border:1px solid #4285F4;">'
        f'<div style="font-size:28px;">🔍</div>'
        f'<div style="font-size:11px;color:#888;margin:4px 0;">RECHERCHE GOOGLE</div>'
        f'<div style="font-size:18px;font-weight:700;color:#4285F4;">{fmt_google(kwh)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="{_card_e}background:#f0faf8;border:1px solid #2A9D8F;">'
        f'<div style="font-size:28px;">🔋</div>'
        f'<div style="font-size:11px;color:#888;margin:4px 0;">RECHARGE iPHONE</div>'
        f'<div style="font-size:18px;font-weight:700;color:#2A9D8F;">{fmt_iphone(kwh)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Mesure réelle d'AI Energy Score, quand le modèle a été mesuré (modèles open-weight seulement)
    measured = comparable[comparable["model"] == selected_model]
    if not measured.empty:
        wh_measured = float(measured["gpu_energy_wh"].iloc[0])
        st.markdown(
            f'<div style="{_card_e}background:#f7f3ff;border:1px solid #7B61FF;">'
            f'<div style="font-size:28px;">🔬</div>'
            f'<div style="font-size:11px;color:#888;margin:4px 0;">MESURÉ PAR AI ENERGY SCORE (GPU H100)</div>'
            f'<div style="font-size:18px;font-weight:700;color:#7B61FF;">{wh_measured:.3f} Wh / requête</div>'
            f'<div style="font-size:11px;color:#888;margin-top:4px;">estimé par ComparIA : {kwh * 1000:.3f} Wh / requête</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"🔬 Pas de mesure AI Energy Score pour ce modèle : seuls {len(comparable)} modèles open-weight ont été mesurés.")

    with st.expander("📐 Méthodologie — Coût énergétique par requête", expanded=False):
        st.markdown("""
        La consommation affichée est `avg_kwh_per_req` — la moyenne du kWh consommé par requête,
        obtenue en divisant `avg_kwh` (kWh moyen par conversation) par `avg_turns` (nombre moyen de tours).

        Les trois références de comparaison :
        - 💡 **Ampoule LED 60W** — 0.009 kWh par heure d'éclairage
        - 🔍 **Recherche Google** — 0.0003 kWh par requête *(source : rapport environnemental Google 2023)*
        - 🔋 **Recharge iPhone 17 Pro** — 0.022 kWh pour une recharge complète

        Ces comparaisons visent à rendre perceptible un coût énergétique difficile à appréhender en kWh bruts.
        """)
    st.markdown("""
**Key insights**
- 💡 Pour le modèle médian, une requête équivaut à environ **4 minutes** d'ampoule LED ; un tiers des modèles reste sous la minute
- 🔍 De 0,07 recherche Google (modèles les plus sobres) à plus de 300 (*grok-4.20*), pour une médiane d'environ 2 recherches
- 🔋 Le modèle médian consomme environ 3 % d'une recharge iPhone par requête ; *grok-4.20* dépasse 4 recharges selon l'estimation de ComparIA (valeur extrême, à interpréter avec prudence)
""")

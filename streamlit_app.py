import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Paths
# -----------------------------
ART_DIR = Path("artifacts")
BUNDLE_PATH = ART_DIR / "model_bundle.joblib"
META_PATH = ART_DIR / "meta.json"


# -----------------------------
# Loaders (cached)
# -----------------------------
@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing: {BUNDLE_PATH}")
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_meta():
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# -----------------------------
# Notebook-style utilities
# -----------------------------
def proba_pos(model, X):
    """Return P(y=1) for models that implement predict_proba."""
    p = model.predict_proba(X)
    return p[:, 1]


def soft_vote(probs_list, weights=None):
    """
    probs_list: list of arrays, each shape (n,)
    weights: None or array-like shape (M,)
    """
    P = np.vstack(probs_list)  # (M, n)
    if weights is None:
        w = np.ones(P.shape[0], dtype=float) / P.shape[0]
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    return (w[:, None] * P).sum(axis=0)


def make_X_t(bundle, preprocess, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw input using preprocess pipeline, then wrap into a DataFrame
    with columns=feature_names_out (exactly like your notebook).
    """
    X_arr = preprocess.transform(X_raw)
    try:
        X_arr = X_arr.toarray()
    except Exception:
        X_arr = np.asarray(X_arr)
    return pd.DataFrame(X_arr, columns=list(bundle["feature_names_out"]))


def predict_proba_bad_ensemble(
    X_raw: pd.DataFrame,
    preprocess,
    ebm_model,
    lr_model,
    ada_model,
    feature_names_out,
    weights=None,
):
    """
    Notebook-equivalent ensemble probability + transformed X_t DataFrame.
    Returns: (p_bad (n,), X_t)
    """
    if weights is None:
        weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    X_arr = preprocess.transform(X_raw)
    try:
        X_arr = X_arr.toarray()
    except Exception:
        X_arr = np.asarray(X_arr)

    X_t = pd.DataFrame(X_arr, columns=list(feature_names_out))

    p_ebm = proba_pos(ebm_model, X_t)
    p_lr = proba_pos(lr_model, X_t)
    p_ada = proba_pos(ada_model, X_t)

    p_ens = soft_vote([p_ebm, p_lr, p_ada], w)
    return p_ens, X_t, (p_ebm, p_lr, p_ada), w


def ebm_deny_reasons(ebm_model, X_t: pd.DataFrame, i=0, top_k=3):
    """
    Your notebook logic:
    - local = ebm_model.explain_local(X_t.iloc[[i]])
    - data0 = local.data(0)
    - names = data0["names"]
    - scores = np.array(data0["scores"])
    - take top_k positive scores (scores > 0), descending
    Returns: list[(term_name, contribution)]
    """
    local = ebm_model.explain_local(X_t.iloc[[i]])
    data0 = local.data(0)
    names = data0["names"]
    scores = np.array(data0["scores"], dtype=float)

    pos_idx = np.argsort(-scores)
    top = []
    for j in pos_idx:
        if scores[j] <= 0:
            continue
        top.append((names[j], float(scores[j])))
        if len(top) >= top_k:
            break
    return top


# -----------------------------
# Natural-language reasons (copy your notebook mapping here)
# -----------------------------
BASE_REASON_MAP = {
    "ExternalRiskEstimate": "Your external risk score appears relatively low.",
    "MSinceMostRecentDelq": "We could not verify recent delinquency timing information from your credit record.",
    "MSinceMostRecentInqexcl7days": "There are signs of frequent recent credit inquiries.",
    "NetFractionRevolvingBurden": "Your revolving credit utilization appears high.",
    "NumBank2NatlTradesWHighUtilization": "Several accounts appear to have high utilization.",
    "NumTrades90Ever2DerogPubRec": "Your credit file indicates prior serious negative events (e.g., derogatory/public records).",
    "NumTotalTrades": "Your overall credit account history/structure appears less favorable.",
    "AverageMInFile": "Your average length of credit history appears shorter or less established.",
    "PercentTradesWBalance": "A high share of accounts appear to carry balances.",
    "NumSatisfactoryTrades": "The number of satisfactory trades appears lower than desired.",
    "MaxDelqEver": "Your history suggests severe delinquency levels at some point.",
    "MaxDelq2PublicRecLast12M": "There are recent indicators of delinquency/public records within the last 12 months.",
    "NumInqLast6M": "Recent credit inquiries in the last 6 months appear elevated.",
    "NumTradesOpeninLast12M": "The number of newly opened accounts in the last 12 months appears elevated.",
}

# Handles indicator terms like "Feature==-7", "Feature==-8", "Feature==-9"
MISSING_PAT = re.compile(r"^(.*)==-([789])$")


def feature_to_plain_reason(term_name: str) -> str:
    """
    Converts a model term name into a plain-English denial reason.
    Matches your notebook behavior:
    - If term looks like "X==-7/-8/-9": treat as missing-code indicator and map by base name
    - Else: map directly or fallback generic sentence
    """
    m = MISSING_PAT.match(term_name)
    if m:
        base = m.group(1)
        if base in BASE_REASON_MAP:
            return BASE_REASON_MAP[base]
        return f"We could not verify some key information related to '{base}'."

    return BASE_REASON_MAP.get(
        term_name,
        f"A key risk factor related to '{term_name}' contributed to this decision."
    )


def format_denial_reasons_english(reasons, top_k=3):
    """
    reasons: list[(term_name, contribution)] sorted desc
    returns: list[dict] with message + term + strength
    """
    out = []
    for term, contrib in reasons[:top_k]:
        out.append(
            {
                "message": feature_to_plain_reason(term),
                "model_term": term,
                "strength": float(contrib),
            }
        )
    return out


# -----------------------------
# UI: build raw input form
# -----------------------------
def build_input_form(input_columns, meta):
    """
    Create numeric inputs for all raw columns (23).
    If you want min/max constraints from meta, we can add them later.
    """
    schema = meta.get("feature_schema", {}) if isinstance(meta, dict) else {}
    x = {}
    for col in input_columns:
        spec = schema.get(col, {})
        default = spec.get("default", 0)

        try:
            dv = float(default)
        except Exception:
            dv = 0.0

        x[col] = float(st.number_input(col, value=dv, format="%.6f"))
    return pd.DataFrame([x])


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Credit Risk Demo", layout="centered")
st.title("Credit Risk Decision Demo (Ensemble + EBM Reasons)")

bundle = load_bundle()
meta = load_meta()

preprocess = bundle["preprocess"]
models = bundle["models"]
weights = np.asarray(bundle.get("weights", [1 / 3, 1 / 3, 1 / 3]), dtype=float)
thr0 = float(bundle.get("thr", 0.5))
input_columns = list(bundle["input_columns"])
feature_names_out = list(bundle["feature_names_out"])

# Sidebar settings
st.sidebar.header("Settings")
thr = st.sidebar.slider("Decision threshold (thr)", 0.0, 1.0, thr0, 0.01)
top_k = st.sidebar.selectbox("Top-K reasons (EBM)", [1, 2, 3, 5, 8], index=2)

st.sidebar.subheader("Ensemble weights (ebm, l1_logreg, adaboost)")
st.sidebar.write(list(weights))

show_model_breakdown = st.sidebar.checkbox("Show model probability breakdown", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# Input
st.subheader("Input (raw features)")
X_raw = build_input_form(input_columns, meta)
st.dataframe(X_raw, use_container_width=True)

# Predict
if st.button("Predict", type="primary"):
    try:
        ebm = models["ebm"]
        lr = models["l1_logreg"]
        ada = models["adaboost"]

        p_bad_arr, X_t, (p_ebm, p_lr, p_ada), w_norm = predict_proba_bad_ensemble(
            X_raw=X_raw,
            preprocess=preprocess,
            ebm_model=ebm,
            lr_model=lr,
            ada_model=ada,
            feature_names_out=feature_names_out,
            weights=weights,
        )

        p_bad = float(p_bad_arr[0])
        pred = int(p_bad >= thr)

        st.metric("P(Bad)", p_bad)
        st.write(f"**Pred (1=Bad/deny, 0=approve preliminarily):** {pred}")

        if pred == 0:
            st.success("Your application has been preliminarily approved.")
        else:
            st.error("We're sorry—your application was not approved at this time.")

        # Reasons (only show denial reasons when pred==1, like notebook)
        st.subheader("Reasons")
        if pred == 1:
            reasons_terms = ebm_deny_reasons(ebm, X_t, i=0, top_k=int(top_k))
            reasons_msgs = format_denial_reasons_english(reasons_terms, top_k=int(top_k))

            if len(reasons_msgs) == 0:
                st.write("No positive (risk-increasing) EBM reasons found for this case.")
            else:
                # Show as numbered readable text + a table
                st.markdown("### Natural-language explanation")
                for idx, r in enumerate(reasons_msgs, 1):
                    st.write(f"{idx}. {r['message']} (term={r['model_term']}, strength={r['strength']:.4f})")

                st.markdown("### Raw EBM terms")
                st.dataframe(pd.DataFrame(reasons_msgs), use_container_width=True)
        else:
            st.write("No denial reasons (approved).")

        if show_model_breakdown:
            st.subheader("Model probability breakdown")
            dfp = pd.DataFrame(
                {
                    "model": ["ebm", "l1_logreg", "adaboost", "ensemble"],
                    "p_bad": [float(p_ebm[0]), float(p_lr[0]), float(p_ada[0]), p_bad],
                    "weight": [float(w_norm[0]), float(w_norm[1]), float(w_norm[2]), 1.0],
                }
            )
            st.dataframe(dfp, use_container_width=True)

        if show_debug:
            st.subheader("Debug")
            st.write("raw shape:", X_raw.shape)
            st.write("processed columns:", len(feature_names_out))
            st.write("X_t shape:", X_t.shape)
            st.write("first 5 processed cols:", feature_names_out[:5])
            st.write("bundle keys:", list(bundle.keys()))

    except Exception as e:
        st.exception(e)

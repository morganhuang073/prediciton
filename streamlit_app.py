import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ART_DIR = Path("artifacts")
BUNDLE_PATH = ART_DIR / "model_bundle.joblib"
META_PATH = ART_DIR / "meta.json"


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing model bundle: {BUNDLE_PATH}\n"
            "Please add artifacts/model_bundle.joblib to this project."
        )
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_meta():
    if not META_PATH.exists():
        # meta 不是必须，但有的话可以用来生成表单/显示信息
        return {}
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def to_dataframe_from_inputs(feature_schema: dict) -> pd.DataFrame:
    """
    feature_schema 期望形如：
    {
      "age": {"type":"int","min":18,"max":100,"default":30},
      "income": {"type":"float","min":0,"max":1000000,"default":50000},
      "job": {"type":"cat","choices":["A","B"],"default":"A"}
    }
    如果 meta.json 没提供 schema，就用一个空 schema（你需要自己补）。
    """
    x = {}
    for name, spec in feature_schema.items():
        t = spec.get("type", "float")
        default = spec.get("default", 0)

        if t == "int":
            x[name] = st.number_input(
                name,
                min_value=int(spec.get("min", -10**9)),
                max_value=int(spec.get("max", 10**9)),
                value=int(default),
                step=1,
            )
        elif t == "float":
            x[name] = st.number_input(
                name,
                min_value=float(spec.get("min", -1e18)),
                max_value=float(spec.get("max", 1e18)),
                value=float(default),
            )
        elif t == "cat":
            choices = spec.get("choices", [])
            if not choices:
                choices = [str(default)]
            x[name] = st.selectbox(name, choices, index=choices.index(default) if default in choices else 0)
        else:
            x[name] = st.text_input(name, value=str(default))

    return pd.DataFrame([x])


def get_p_bad(model, X: pd.DataFrame) -> float:
    """
    尝试用 predict_proba 得到坏客户概率。
    若你的正类不是 1，需要按你训练时的类标签调整。
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # 默认取第二列 = class 1 概率
        return float(proba[0, 1])
    # 兜底：只有 predict 时，假设输出已是概率
    pred = model.predict(X)
    return float(pred[0])


st.set_page_config(page_title="Bank Decision App", layout="centered")
st.title("Bank Decision App")

bundle = load_bundle()
meta = load_meta()

# 你之前的 bundle 看起来像是：包含 models、feature_names_out、weights、thr 等
models = bundle.get("models", {})
thr = bundle.get("thr", 0.5)
feature_names_out = bundle.get("feature_names_out", None)
weights = bundle.get("weights", None)

st.sidebar.header("Settings")
thr = st.sidebar.slider("Decision threshold (thr)", 0.0, 1.0, float(thr), 0.01)

model_name = st.sidebar.selectbox(
    "Choose model",
    options=list(models.keys()) if models else ["(no models found in bundle)"],
)

st.subheader("Input")
feature_schema = meta.get("feature_schema", {})  # 如果你 meta.json 没有这个字段，就会是 {}
if not feature_schema:
    st.info(
        "meta.json 没提供 feature_schema，无法自动生成输入表单。\n"
        "你需要在 artifacts/meta.json 里加入 feature_schema，或在代码里手动写输入项。"
    )

X = to_dataframe_from_inputs(feature_schema) if feature_schema else pd.DataFrame([{}])
st.write("Your input row:", X)

if st.button("Predict", type="primary", disabled=(not models or model_name not in models or X.empty)):
    model = models[model_name]
    p_bad = get_p_bad(model, X)

    pred = int(p_bad >= thr)

    st.metric("P(Bad)", p_bad)
    st.write(f"**Pred (1=Bad/deny, 0=approve preliminarily):** {pred}")

    if pred == 0:
        st.success("Your application has been preliminarily approved.")
    else:
        st.error("We're sorry—your application was not approved at this time.")

    # 可选：展示权重/特征重要性（如果 bundle 里有）
    if feature_names_out is not None and weights is not None:
        try:
            dfw = pd.DataFrame({"feature": feature_names_out, "weight": weights})
            dfw["abs_weight"] = dfw["weight"].abs()
            st.subheader("Feature weights")
            st.dataframe(dfw.sort_values("abs_weight", ascending=False).drop(columns=["abs_weight"]), use_container_width=True)
        except Exception as e:
            st.caption(f"Could not render weights: {e}")
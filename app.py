"""Fast per-pair inference server for the OYNX / Quotex trading agent.

No training happens here. Every model produced by train_agent.py is loaded once
at startup (one model per asset + timeframe) and each request is routed to the
model for the asset and timeframe the caller sent.

Model files are discovered by the naming convention:

    trained_{asset}_{timeframe}.json      e.g. trained_EURUSD_M1.json
                                               trained_USDBDT_otc_M1.json

Run:
    python -m uvicorn app:app --host 0.0.0.0 --port 8000
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_TEMPLATE = "trained_*_*.json"

FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_30",
    "rsi_14",
    "atr_pct",
    "roc_5",
    "macd_pct",
    "macd_hist_pct",
    "macd_signal_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_pct",
    "vol_rel",
]

# MACD needs 26 + 9 candles to stabilise, rolling windows need 30 -> 50 is safe
MIN_CANDLES = 50

app = FastAPI(title="OYNX Market Analysis API")

# The Lovable bot calls this from a browser, so cross-origin requests must be allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The API only keeps a ~10-minute ring buffer for these, so no model can be
# trained on them (30s has as few as 16 candles total). Requests for them are
# served by the pair's M1 model instead, and the response says so.
SUB_MINUTE_TIMEFRAMES = {"5s", "10s", "15s", "30s"}
FALLBACK_TIMEFRAME = "M1"

# The models are trained on triple-barrier labels, and a 1.5xATR take-profit is
# further away than the 1.0xATR stop, so only ~28% of candles are winners. The
# probabilities therefore sit near each pair's own base rate (~0.28) instead of
# 0.50, and fixed 0.75/0.25 cut-offs would almost never fire. Each model's base
# rate is read from its own file, and a signal requires this much edge over it.
SIGNAL_EDGE = 0.12

# (asset, timeframe) -> (model, the base rate it was trained on)
MODELS: dict[tuple[str, str], tuple[xgb.XGBClassifier, float]] = {}


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class PredictRequest(BaseModel):
    asset: str
    timeframe: str = "M1"
    candles: list[Candle] = Field(min_length=MIN_CANDLES)


class PredictResponse(BaseModel):
    # "model_timeframe" starts with "model_", which Pydantic reserves by default
    model_config = {"protected_namespaces": ()}

    asset: str
    timeframe: str
    signal: str
    confidence_score: str
    # Which model actually produced this signal. Differs from `timeframe` when a
    # sub-minute request was served by the pair's M1 model (see SUB_MINUTE_TIMEFRAMES).
    model_timeframe: str


def parse_model_filename(path):
    """trained_USDBDT_otc_M1.json -> ('USDBDT_otc', 'M1').

    Assets may contain underscores, timeframes never do, so the timeframe is
    whatever follows the final underscore.
    """
    stem = os.path.basename(path)[len("trained_"): -len(".json")]
    asset, _, timeframe = stem.rpartition("_")
    return asset, timeframe


def discover_models():
    """Map (asset, timeframe) -> model file path for everything trained so far."""
    found = {}
    for path in sorted(glob.glob(MODEL_TEMPLATE)):
        asset, timeframe = parse_model_filename(path)
        if asset and timeframe:
            found[(asset, timeframe)] = path
    return found


def read_model(path):
    """Load a model and the base rate it was trained on.

    XGBoost stores the mean of the training labels as `base_score`, which for
    these triple-barrier labels is the pair's win rate. That is where "no edge"
    sits for this pair, and the signal thresholds are placed relative to it.
    """
    model = xgb.XGBClassifier()
    model.load_model(path)

    config = json.loads(model.get_booster().save_config())
    base_rate = float(config["learner"]["learner_model_param"]["base_score"])
    return model, base_rate


def load_models():
    """Load every trained_*.json next to this file, keyed by (asset, timeframe)."""
    return {key: read_model(path) for key, path in discover_models().items()}


def get_model(asset, timeframe):
    """Return the model for a pair, reading it from disk the first time it is asked for.

    Training runs in the background, so model files keep appearing while this
    server is already up. Loading on demand means new pairs are served without
    a restart.
    """
    key = (asset, timeframe)
    if key not in MODELS:
        path = discover_models().get(key)
        if path is None:
            return None
        MODELS[key] = read_model(path)
        print(f"Loaded {asset}@{timeframe} on demand")
    return MODELS[key]


MODELS = load_models()
if not MODELS:
    raise RuntimeError(
        "No trained_*.json models found. Run 'python train_agent.py "
        "--assets USDBDT_otc,EURUSD,EURGBP --timeframe M1' first, and make sure "
        "the model files are deployed alongside app.py."
    )
print(f"Loaded {len(MODELS)} model(s): {', '.join(f'{a}@{t}' for a, t in MODELS)}")


def build_features(df):
    """Compute the exact feature set the models were trained on."""
    df = df.copy()
    close = df["close"]

    df["ret_1"] = np.log(close / close.shift(1))
    df["ret_3"] = np.log(close / close.shift(3))
    df["ret_5"] = np.log(close / close.shift(5))
    df["ret_10"] = np.log(close / close.shift(10))
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()

    df["rsi_14"] = ta.rsi(close, length=14)
    atr = ta.atr(df["high"], df["low"], close, length=14)
    df["atr_pct"] = atr / close
    df["roc_5"] = ta.roc(close, length=5)

    macd = ta.macd(close)
    df["macd_pct"] = macd.iloc[:, 0] / close
    df["macd_hist_pct"] = macd.iloc[:, 1] / close
    df["macd_signal_pct"] = macd.iloc[:, 2] / close

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    df["body_pct"] = (close - df["open"]) / candle_range
    df["upper_wick_pct"] = (df["high"] - body_high) / candle_range
    df["lower_wick_pct"] = (body_low - df["low"]) / candle_range
    df["range_pct"] = candle_range / close

    volume_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["vol_rel"] = df["volume"] / volume_ma
    return df


@app.get("/")
def root():
    """Browser-friendly status page - open http://127.0.0.1:8000/ to check it is live."""
    pairs = sorted(discover_models())
    return {
        "status": "online",
        "models_loaded": len(pairs),
        "pairs": [f"{a}@{t}" for a, t in pairs],
        "endpoints": {"predict": "POST /analyze-and-train", "models": "GET /models"},
    }


@app.get("/models")
def list_models():
    """Which pairs/timeframes this server can serve."""
    return {"models": [{"asset": a, "timeframe": t} for a, t in sorted(discover_models())]}


@app.get("/analyze-and-train")
def analyze_help():
    """Opening this URL in a browser sends GET, but the real endpoint is POST.

    Without this route FastAPI answers a confusing 405 Method Not Allowed.
    """
    return {
        "detail": "This endpoint only accepts POST.",
        "method": "POST",
        "content_type": "application/json",
        "example_body": {
            "asset": "BTCUSD_otc",
            "timeframe": "M1",
            "candles": [
                {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 10}
            ],
        },
        "candles_required": MIN_CANDLES,
        "note": (
            f"Send at least {MIN_CANDLES} closed candles, oldest first. "
            "Call GET /models for the list of available asset/timeframe pairs."
        ),
    }


@app.post("/analyze-and-train", response_model=PredictResponse)
def analyze(payload: PredictRequest):
    entry = get_model(payload.asset, payload.timeframe)
    model_timeframe = payload.timeframe

    # No model can be trained for 5s/10s/15s/30s, so those fall back to M1
    if entry is None and payload.timeframe in SUB_MINUTE_TIMEFRAMES:
        entry = get_model(payload.asset, FALLBACK_TIMEFRAME)
        if entry is not None:
            model_timeframe = FALLBACK_TIMEFRAME

    if entry is None:
        available = ", ".join(f"{a}@{t}" for a, t in sorted(discover_models()))
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trained model for {payload.asset} @ {payload.timeframe}. "
                f"Available: {available}. "
                f"Train it with: python train_agent.py --assets {payload.asset} "
                f"--timeframe {model_timeframe}"
            ),
        )

    df = pd.DataFrame([c.model_dump() for c in payload.candles])
    df = build_features(df)

    # Only the latest fully closed candle is scored
    latest = df[FEATURES].iloc[[-1]]

    if latest.isna().values.any():
        missing = [c for c in FEATURES if latest[c].isna().any()]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Latest candle produced NaN features ({', '.join(missing)}). "
                f"Send at least {MIN_CANDLES} closed candles so the indicators warm up."
            ),
        )

    model, base_rate = entry
    probability = float(model.predict_proba(latest)[0][1])

    if probability >= base_rate + SIGNAL_EDGE:
        signal = "STRONG BUY"
    elif probability <= base_rate - SIGNAL_EDGE:
        signal = "STRONG SELL"
    else:
        signal = "NEUTRAL"

    return PredictResponse(
        asset=payload.asset,
        timeframe=payload.timeframe,
        signal=signal,
        confidence_score=f"{probability * 100:.2f}%",
        model_timeframe=model_timeframe,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

#!/usr/bin/env python3
"""FastAPI microservice for Iris class prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from lab3.model import TARGET_NAMES, load_model

MODEL = None


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    MODEL = load_model()
    yield


app = FastAPI(title="Lab 3 Iris Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    features = np.array(
        [[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]],
        dtype=float,
    )
    probabilities = MODEL.predict_proba(features)[0]
    predicted_index = int(np.argmax(probabilities))

    return PredictResponse(
        prediction=TARGET_NAMES[predicted_index],
        probabilities={
            TARGET_NAMES[index]: round(float(score), 4)
            for index, score in enumerate(probabilities)
        },
    )

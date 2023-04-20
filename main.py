from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np

import openai
from openai.embeddings_utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from os import getenv


def load_data() -> pd.DataFrame:
    df = pd.read_pickle("data/EmbeddingsEmissionFactorsData.pkl")
    return df


load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

app = FastAPI()
app.state.dataset = load_data()


@app.get("/")
def home() -> dict:
    """
    Homepage
    """

    return {
        "status": "The emissions service is running!",
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }


class EmissionItem(BaseModel):
    """
    Emission item
    """

    item: str
    emissions_factor: float
    confidence: float


class EmissionResult(BaseModel):
    """
    An emission result with the top 5 most similar emission items
    """

    desc: str
    emission_item: list[EmissionItem]


def clean(text: str) -> str:
    """
    Change to lowercase and remove all special characters
    """
    text = text.casefold()
    text = text.replace(r"[^a-zA-Z0-9\s]", "")
    return text


@app.post("/predict")
async def predict(descriptions: list[str]) -> list[EmissionResult]:
    """
    Predicts the top 5 emissions for a list of input item descriptions
    with a confidence threshold
    """

    openai.api_key = OPENAI_API_KEY
    embedding_model = "text-embedding-ada-002"

    descriptions = [clean(desc) for desc in descriptions]
    embeddings = [get_embedding(desc, embedding_model) for desc in descriptions]

    dataset = app.state.dataset
    confidence_threshold = 0.8

    cosine_similarities = cosine_similarity(
        np.asarray(embeddings), np.asarray(dataset["EMBEDDING"].values.tolist())
    )

    results: list[EmissionResult] = []
    for desc, sim in zip(descriptions, cosine_similarities):
        top_indices = sim.argsort()[-5:][::-1]
        results.append(
            EmissionResult(
                desc=desc,
                emission_item=[
                    EmissionItem(
                        item=dataset.iloc[i]["ITEM"],
                        emissions_factor=dataset.iloc[i]["EMISSIONS_FACTOR"],
                        confidence=sim[i],
                    )
                    for i in top_indices
                    if sim[i] > confidence_threshold
                ],
            )
        )

    return results

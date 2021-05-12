"""Tests for web app."""
import base64
import os
from pathlib import Path

from fastapi.testclient import TestClient

from api_server.main import app

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_DIRNAME = Path(__file__).parents[2].resolve()
SUPPORT_DIRNAME = REPO_DIRNAME / "im2latex" / "tests" / "support" / "im2latex_100k"
FILENAME = SUPPORT_DIRNAME / "7944775fc9.png"
EXPECTED_PRED = "\\alpha _ { 1 } ^ { \\gamma } \\gamma _ { 1 } + . . . + \\alpha _ { N } ^ { \\gamma } \\gamma _ { N } = 0 \\quad ( r = 1 , . . , R ) \\, ,"

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_predict():
    with open(FILENAME, "rb") as f:
        b64_image = base64.b64encode(f.read())
    response = client.post("/v1/predict", json={"image": f"data:image/png;base64,{b64_image.decode()}"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data["pred"] == EXPECTED_PRED

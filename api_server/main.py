import logging
import os

import uvicorn
from fastapi import FastAPI, status
from fastapi.exceptions import HTTPException
from PIL import ImageStat
from pydantic import BaseModel

import im2latex.util as util
from im2latex.im2latex_inference import Im2LatexInference

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU


class ImageRequest(BaseModel):
    image: str


class LatexResponse(BaseModel):
    pred: str


model = Im2LatexInference()
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/v1/predict", response_model=LatexResponse, status_code=status.HTTP_200_OK)
def get_predict(image_url: str = None):
    if image_url is None or not image_url.startswith("http"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="no image_url defined in query string")
    # logging.info("url {}".format(image_url))
    logging.info("url %s", image_url)
    image = util.read_image_pil(image_url, grayscale=True)
    pred = model.predict(image)
    image_stat = ImageStat.Stat(image)
    # logging.info("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    # logging.info("METRIC image_area {}".format(image.size[0] * image.size[1]))
    # logging.info("METRIC pred_length {}".format(len(pred)))
    # logging.info("pred {}".format(pred))
    logging.info("METRIC image_mean_intensity %f", image_stat.mean[0])
    logging.info("METRIC image_area %d", image.size[0] * image.size[1])
    logging.info("METRIC pred_length %d", len(pred))
    logging.info("pred %s", pred)
    return {"pred": str(pred)}


@app.post("/v1/predict", response_model=LatexResponse, status_code=status.HTTP_200_OK)
async def post_predict(json_request: ImageRequest):
    if not json_request.image.startswith("data:image/png;base64,"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Add 'data:image/png;base64,' before base64 encoded string"
        )
    image = util.read_b64_image(json_request.image, grayscale=True)
    pred = model.predict(image)
    image_stat = ImageStat.Stat(image)
    # logging.info("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    # logging.info("METRIC image_area {}".format(image.size[0] * image.size[1]))
    # logging.info("METRIC pred_length {}".format(len(pred)))
    # logging.info("pred {}".format(pred))
    logging.info("METRIC image_mean_intensity %f", image_stat.mean[0])
    logging.info("METRIC image_area %d", image.size[0] * image.size[1])
    logging.info("METRIC pred_length %d", len(pred))
    logging.info("pred %s", pred)
    return {"pred": str(pred)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

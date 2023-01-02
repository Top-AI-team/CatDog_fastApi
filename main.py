from fastapi import FastAPI, File, Form, Request
import uvicorn
import requests

import model


app = FastAPI()


@app.get("/api/v1/predict/")
async def root(img: bytes | None = File(default=None), url: str | None = Form(default=None)):
    if not img:
        if not url:
            return {"message": "No data sent"}
        img = requests.get(url).content
    return {"prediction": model.predict_class(img)}


@app.get("/api/v1/predict_many/")
async def root(request: Request):
    url_list = await request.json()
    if not url_list:
        return {"message": "No data sent"}

    res = []

    for url in url_list:
        img = requests.get(url).content
        print(len(img))
        res.append(model.predict_class(img))
    return {
        "cats_count": res.count("Cat"),
        "dogs_count": res.count("Dog"),
        "data_res": res
    }


if __name__ == "__main__":
    uvicorn.run(app)

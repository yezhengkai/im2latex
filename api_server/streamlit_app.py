import base64
import socket
import subprocess
from time import sleep

import requests
import streamlit as st

API_HOST = "0.0.0.0"
API_PORT = 60000
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
API_PREDICT_URL = f"{API_BASE_URL}/v1/predict"


# Reference: https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


# Reference: https://github.com/thoppe/streamlit-CLIP-Unsplash-explorer/blob/master/start_api.py
def start_api():
    if not is_port_in_use(API_PORT):
        with st.spinner("Starting the API..."):
            print("Starting the API")
            cmd = ["uvicorn", "api_server.main:app", "--host", f"{API_HOST}", "--port", f"{API_PORT}"]
            subprocess.Popen(cmd, close_fds=True)
            sleep(5)


def main():
    st.title("Image to LaTeX")

    with st.form(key="imputs"):
        st.markdown("## Upload an Image")
        image = st.file_uploader("", type=["jpg", "png"])
        st.form_submit_button(label="Upload")

    with st.form(key="outputs"):
        st.markdown("## Convert to LaTeX")
        st.text("Uploaded image:")
        if image is not None:
            st.image(image)
        inference_button = st.form_submit_button(label="Infer")

    if inference_button and image is not None:
        with st.spinner("Wait for it..."):
            b64_image = base64.b64encode(image.read())
            response = requests.post(API_PREDICT_URL, json={"image": f"data:image/png;base64,{b64_image.decode()}"})
        if response.status_code == 200:
            st.markdown("**Inferred LaTex:**")
            st.code(response.json()["pred"], language="latex")
            st.markdown("**Render Inferred LaTeX:**")
            st.latex(response.json()["pred"])
        else:
            st.error("An error occurred in the model server.")


if __name__ == "__main__":
    start_api()
    main()

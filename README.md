# Image to Latex
Convert the image of the formula to Latex.

# Training
## Run experiment
Under the project root directory, run 
```bash
python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=2 --model_class=ResnetTransformer --data_class=Im2Latex100K --batch_size=16
```

Use the `wandb init` command to set up a new W&B project, so we can add `--wandb` to record our experiments through the service provided by W&B.
```bash
python training/run_experiment.py --wandb --max_epochs=3 --gpus='0,' --num_workers=2 --model_class=ResnetTransformer --data_class=Im2Latex100K --batch_size=8
```

If you want to test your model you can add `--overfit_batches` argument.
For more argument usage, you can refer to [pytorch-lightning Trainer](https://pytorch-lightning.readthedocs.io/en/1.2.8/common/trainer.html).

### Supported model_class
- CNNLSTM
- ResnetTransformer

### data_class
- Im2Latex100K

## Save best model
Under the project root directory, run 
```bash
python training/save_best_model.py --entity=zhengkai --project=im2latex --trained_data_class=Im2Latex100K
```

- `--entity`: your W&B user name
- `--project`: your W&B project

# Inference
Under the project root directory, run `python im2latex/im2latex_inference.py <image_path>`, for example:
```bash
python im2latex/im2latex_inference.py im2latex/tests/support/im2latex_100k/7944775fc9.png
```

# Serving model
## Build the image
Under the project root directory, run 
```bash
docker build -t im2latex/api-server -f api_server/Dockerfile .
```

If you want to rebuild the image, you can use the following command to remove the existing image.
```bash
docker rmi -f im2latex/api-server
```

## Run the container
Under the project root directory, run
```bash
docker run -p 60000:60000 -p 60001:60001 -it --rm --name im2latex-api im2latex/api-server
```
Then, we can use the model API through port 60000 and use the Streamlit App through port 60001.

If the container is already running, you can use the following command to remove the existing container.
```bash
docker rm -f im2latex-api
```

# Code test
## Inference tests
Under the project root directory, run
```bash
pytest -s ./im2latex/tests/test_im2latex_inference.py
```

## Evaluation tests
Under the project root directory, run
```bash
pytest -s ./im2latex/evaluation/evaluate_im2latex_inference.py
```

## API server tests
Under the project root directory, run
```bash
pytest -s api_server/tests/test_app.py
```

# Streamlit App
You can try [Image to LaTeX App](https://share.streamlit.io/yezhengkai/im2latex/api_server/streamlit_app.py) online. But please note that for images other than the training data set, the model performance is still very poor.

# References
- [luopeixiang/im2latex](https://github.com/luopeixiang/im2latex)
- [fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs)
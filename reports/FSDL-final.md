- [Full Stack Deep Learning Final Project - im2latex](#full-stack-deep-learning-final-project---im2latex)
  - [Project Description](#project-description)
  - [My main goal](#my-main-goal)
  - [Implements](#implements)
    - [Environment/Package Management](#environmentpackage-management)
    - [Linter and Formatter](#linter-and-formatter)
    - [Data Management](#data-management)
    - [Model Architecture](#model-architecture)
    - [Train](#train)
    - [Evaluate](#evaluate)
    - [Deployment](#deployment)
    - [Code Test](#code-test)
  - [Results](#results)
    - [Example 1](#example-1)
    - [Example 2](#example-2)
    - [Example 3](#example-3)
  - [Conclusion](#conclusion)


# Full Stack Deep Learning Final Project - im2latex

## Project Description
This project is the final project of Full Stack Deep Learning Spring 2021. The main objective of the im2latex project is to convert the image of the formula to Latex.

## My main goal
Through this project, I want to realize the workflow that I learned in the fsdl course. Therefore, my focus is on how to build an integrated ML project from scratch, and mainly refer to the [full-stack-deep-learning/fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) and [luopeixiang/im2latex](https://github.com/luopeixiang/im2latex) repositories.

## Implements
This section will describe the details of my implementation and the tools used.

### Environment/Package Management
- anaconda
- shell (cmd in Windows)

The FSDL course uses `conda` and `pip-tools` to manage packages, which is a good solution to reproduce the same environment on different machines. But I think `conda` has provided a convenient way to deal with this problem, so I only use `conda` to set up my environment. I just need to add comments in "environment.yml" to mark the package as development/production, so that I can easily create a "requirement.txt" for pip for development/production.

In addition, since the OS of the local machine is Windows, I wrote a batch file to imitate make in Linux. So far, the batch file has worked well.

### Linter and Formatter
- pycodestyle
- pylint
- pydocstyle
- mypy
- bandit
- safety
- black
- isort

Linters are a good tool for finding potential errors in the code, and `mypy` can help us complete static type hinting, which is essential for compiling models into [Torchscript](https://pytorch.org/docs/stable/jit.html).

If the team develops the code together, the formatter can easily unify the code style.

### Data Management
- pytorch + pytorch-lightning

I created a subclass of `Im2Latex100K` inheriting `LightningDataModule`, which sets the workflow of downloading, preprocessing and loading data. When we create an instance of `Im2Latex100K`, if there is no data, the program will download IM2LATEX-100K data from http://lstm.seas.harvard.edu/latex/data/. Then, create a vocabulary (exclude string tokens that appear less than 10 times) and save it to the json file, and dump the image and Latex to the pickle file to reduce loading time.

In order to read images of different sizes in batches, I created a `BucketBatchSampler` to collect images of the same size into one batch. For more detailed information about `BucketBatchSampler`, you can refer to [Tensorflow-esque bucket by sequence length](https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13).

### Model Architecture
- pytorch + pytorch-lightning

Since I am not familiar with the seq2seq model, I first adjusted the seq2seq (CNN encoder + LSTM decoder with Attention) model from the [luopeixiang/im2latex](https://github.com/luopeixiang/im2latex) repository. 
The CNN + LSTM model is slightly different from the original [paper](https://arxiv.org/pdf/1609.04938v1.pdf), and the LSTM part makes its training slower. Therefore, I turned to using the `ResnetTransformer` from the [full-stack-deep-learning/fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) repository.

### Train
- pytorch + pytorch-lightning
- Weights & Biases

With pytorch-lightning, we can conveniently organize model training/validation/test step ([im2latex/lit_models/base.py](../im2latex/lit_models/base.py)), and integrate CLI arguments, Weights & Biases logger, etc. ([training/run_experiment.py](../training/run_experiment.py)). Then, use `wandb sweep training/sweeps/im2latex_100k_resnet_transformer.yml` to create hyperparameter search session on the Weights & Biases. The "Sweep" results can be found in my [**W&B dashboard**](https://wandb.ai/zhengkai/im2latex). We can see the details of each run with different hyperparameter combinations. From the "Sweep" results, the performance of the large model is better than that of the small model. The decrease of "val_loss" each run follows a similar pattern, which shows that we set the random seed correctly. What surprised me during the "Sweep" was that the Bayesian hyperparameter search process used the same parameters at sweep-5 and sweep-6. Maybe random search can prevent similar situations.

I also tried the service provided by [grid.ai](https://www.grid.ai/), but I need to change the existing code or folder structure to use it. Due to time constraints, only Weights & Biases is used at this stage.

### Evaluate
- pytorch + pytorch-lightning

Note that I evaluated the text metrics in the entire test dataset, so the evaluation process may be different from other implementations. And I only use greedy decoding, so the result may not be as good as beam search decoding.

| Model                                                                      | BLEU  | Edit Distance |
| -------------------------------------------------------------------------- | ----- | ------------- |
| [Paper's WYGIWYS](https://arxiv.org/pdf/1609.04938v1.pdf)                  | 87.73 |               |
| [luopeixiang/im2latex's CNN+LSTM](https://github.com/luopeixiang/im2latex) | 40.80 | 44.23         |
| ResnetTransformer                                                          | 59.51 | 65.95         |

In the BLEU metric, our model reached 68% of the performance of the paper's model.

### Deployment
- pytorch + pytorch-lightning
- fastapi
- uvicorn
- docker

For deployment, I referred to the implementation of [full-stack-deep-learning/fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) and changes `flask` to `fastapi + uvicorn`.
In order to make our api server easy to deploy to other machines, here we use docker to create a container to run our api server. To speed up the inference time, we also converted pytorch model to [TorchScript](https://pytorch.org/docs/stable/jit.html).

Server provide a GET method in the "/v1/predict" path so that users can send "image_url" query parameter. Then, server will read image from the URL and return json with `pred` key. Users can also upload base64-encoded images to the server through POST requests.

### Code Test
- pytest

I wrote [inference tests](../im2latex/tests/test_im2latex_inference.py), [evaluation tests](../im2latex/evaluation/evaluate_im2latex_inference.py) and [api tests](../api_server/tests/test_app.py) to test whether the inferred Latex and evaluation metrics meet our expectations and the server is working properly.

## Results
Now we test our model with images from test dataset.

### Example 1
- Input Image:

  <img src=./images/7944775fc9.png alt="\alpha _ { 1 } ^ { r } \gamma _ { 1 } + \dots + \alpha _ { N } ^ { r } \gamma _ { N } = 0 \quad ( r = 1 , . . . , R ) \; ," style="display:block; margin:auto;" />

- Target Latex:
  ```
  \alpha _ { 1 } ^ { r } \gamma _ { 1 } + \dots + \alpha _ { N } ^ { r } \gamma _ { N } = 0 \quad ( r = 1 , . . . , R ) \; ,
  ```

- Inferred Latex:
  ```
  \alpha _ { 1 } ^ { \gamma } \gamma _ { 1 } + . . . + \alpha _ { N } ^ { \gamma } \gamma _ { N } = 0 \quad ( r = 1 , . . , R ) \, ,
  ```

- Render Inferred Latex
  
  <img src="https://latex.codecogs.com/svg.latex?\alpha%20_%20{%201%20}%20^%20{%20\gamma%20}%20\gamma%20_%20{%201%20}%20+%20.%20.%20.%20+%20\alpha%20_%20{%20N%20}%20^%20{%20\gamma%20}%20\gamma%20_%20{%20N%20}%20=%200%20\quad%20(%20r%20=%201%20,%20.%20.%20,%20R%20)%20\,%20," alt="\alpha _ { 1 } ^ { \gamma } \gamma _ { 1 } + . . . + \alpha _ { N } ^ { \gamma } \gamma _ { N } = 0 \quad ( r = 1 , . . , R ) \, ," style="display:block; margin:auto;" />


### Example 2
- Input Image:

  <img src=./images/566cf0c6f5.png alt="\dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) } ; ~ ~ ~ \dot { z } _ { 2 } = - \frac { z _ { 2 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) }"  style="display:block; margin:auto;" />

- Target Latex:
  ```
  \dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) } ; ~ ~ ~ \dot { z } _ { 2 } = - \frac { z _ { 2 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) }
  ```

- Inferred Latex:
  ```
  \dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { z _ { 2 } ( z _ { 2 } - z _ { 1 } ) } ; \quad \dot { z } _ { 2 } = - \frac { z _ { 2 } } { \bar { z } _ { z } ( z _ { 2 } - z _ { 1 } ) }
  ```

- Render Inferred Latex:

  <img src="https://latex.codecogs.com/svg.latex?\dot%20{%20z%20}%20_%20{%201%20}%20=%20-%20N%20^%20{%20z%20}%20(%20z%20_%20{%201%20}%20)%20=%20-%20g%20(%20z%20_%20{%201%20}%20)%20=%20-%20\frac%20{%20z%20_%20{%201%20}%20}%20{%20z%20_%20{%202%20}%20(%20z%20_%20{%202%20}%20-%20z%20_%20{%201%20}%20)%20}%20;%20\quad%20\dot%20{%20z%20}%20_%20{%202%20}%20=%20-%20\frac%20{%20z%20_%20{%202%20}%20}%20{%20\bar%20{%20z%20}%20_%20{%20z%20}%20(%20z%20_%20{%202%20}%20-%20z%20_%20{%201%20}%20)%20}" alt="\dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { z _ { 2 } ( z _ { 2 } - z _ { 1 } ) } ; \quad \dot { z } _ { 2 } = - \frac { z _ { 2 } } { \bar { z } _ { z } ( z _ { 2 } - z _ { 1 } ) }" style="display:block; margin:auto;" />


### Example 3
- Input Image:

  <img src=./images/4c0185889d.png alt="\dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) } ; ~ ~ ~ \dot { z } _ { 2 } = - \frac { z _ { 2 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) }"  style="display:block; margin:auto;" />

- Target Latex:
  ```
  { \cal L } ( J ) = \frac { 1 } { 2 } \partial _ { \mu } \phi \partial ^ { \mu } \phi + \frac { J } { 2 } \phi ^ { 2 } + \frac { \lambda \mu ^ { 2 \varepsilon } } { 4 ! } \phi ^ { 4 } + { \cal L } _ { \mathrm { C T } } ( J ) - \mu ^ { - 2 \varepsilon } \frac { \zeta } { 2 } \; J ^ { 2 } .
  ```

- Inferred Latex:
  ```
  { \cal L } ( J ) = \frac { 1 } { 2 } \partial _ { \mu } \phi \partial ^ { \mu } \phi + \frac { 1 } { 2 } \phi ^ { 2 } + \frac { \lambda \mu ^ { 2 } } { 4 ! } \phi ^ { 4 } + { \cal L } _ { \mathrm { C T } } ( J ) - \mu ^ { 2 } \frac { \xi } { 2 } \, J ^ { 2 } .
  ```

- Render Inferred Latex:

  <img src="https://latex.codecogs.com/svg.latex?{%20\cal%20L%20}%20(%20J%20)%20=%20\frac%20{%201%20}%20{%202%20}%20\partial%20_%20{%20\mu%20}%20\phi%20\partial%20^%20{%20\mu%20}%20\phi%20+%20\frac%20{%201%20}%20{%202%20}%20\phi%20^%20{%202%20}%20+%20\frac%20{%20\lambda%20\mu%20^%20{%202%20}%20}%20{%204%20!%20}%20\phi%20^%20{%204%20}%20+%20{%20\cal%20L%20}%20_%20{%20\mathrm%20{%20C%20T%20}%20}%20(%20J%20)%20-%20\mu%20^%20{%202%20}%20\frac%20{%20\xi%20}%20{%202%20}%20\,%20J%20^%20{%202%20}%20." alt="{ \cal L } ( J ) = \frac { 1 } { 2 } \partial _ { \mu } \phi \partial ^ { \mu } \phi + \frac { 1 } { 2 } \phi ^ { 2 } + \frac { \lambda \mu ^ { 2 } } { 4 ! } \phi ^ { 4 } + { \cal L } _ { \mathrm { C T } } ( J ) - \mu ^ { 2 } \frac { \xi } { 2 } \, J ^ { 2 } ." style="display:block; margin:auto;" />


## Conclusion

The im2latex prototype workflow has been completed at this stage, but there is still a lot of work to be done. For example, use data version control tools (DVC), integrate CI/CD (Github Actions), using a more appropriate model (forcing the model to return paired brackets) and create web app, etc. In any case, we have completed the minimum requirements for machine learning products.

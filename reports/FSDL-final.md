- [Full Stack Deep Learning Final Project - im2latex](#full-stack-deep-learning-final-project---im2latex)
  - [Project Description](#project-description)
  - [My main goal](#my-main-goal)
  - [Implements](#implements)
    - [Environment/Package Management](#environmentpackage-management)
    - [Linter and Formatter](#linter-and-formatter)
    - [Data Management](#data-management)
    - [Model Architecture](#model-architecture)
    - [Training](#training)
    - [Deployment](#deployment)
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

Linters are a good tool for finding potential errors in the code, and `mypy` can help us complete static type hinting, which is essential for compiling models into Torchscript.

If the team develops the code together, the formatter can easily unify the code style.


### Data Management
- pytorch + pytorch-lightning

I created a subclass of `Im2Latex100K` inheriting `LightningDataModule`, which sets the workflow of downloading, preprocessing and loading data. When we create an instance of `Im2Latex100K`, if there is no data, the program will download IM2LATEX-100K data from http://lstm.seas.harvard.edu/latex/data/. Then, create a vocabulary (exclude string tokens that appear less than 10 times) and save it to the json file, and dump the image and Latex to the pickle file to reduce loading time.

In order to read images of different sizes in batches, I created a `BucketBatchSampler` to collect images of the same size into one batch. For more detailed information about `BucketBatchSampler`, you can refer to [Tensorflow-esque bucket by sequence length](https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13).

### Model Architecture
- pytorch + pytorch-lightning

Since I am not familiar with the seq2seq model, I first adjusted the seq2seq (CNN encoder + LSTM decoder with Attention) model from the [luopeixiang/im2latex](https://github.com/luopeixiang/im2latex) repository. 
The CNN + LSTM model is slightly different from the original [paper](http://lstm.seas.harvard.edu/latex/), and the LSTM part makes its training slower. Therefore, I turned to using the ResnetTransformer from the [full-stack-deep-learning/fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) repository.


### Training
- pytorch + pytorch-lightning
- Weights & Biases

I also tried the service provided by [grid.ai](https://www.grid.ai/), but I need to change the existing code or folder structure to use it. Due to time constraints, only Weights & Biases is used at this stage.

### Deployment
- fastapi
- uvicorn
- docker

For deployment, I referred to the implementation of[full-stack-deep-learning/fsdl-text-recognizer-2021-labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs) and changes `flask` to `fastapi + uvicorn`.
In order to make our api server easy to deploy to other machines, I used docker to create a container to run our api server.


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
  \alpha_{r}^{\prime}\gamma_{1}+\ldots+\alpha_{r}^{\prime}\gamma_{N}=0\quad(r=1,\ldots,R)\;,
  ```

- Render Inferred Latex
  $$\alpha_{r}^{\prime}\gamma_{1}+\ldots+\alpha_{r}^{\prime}\gamma_{N}=0\quad(r=1,\ldots,R)\;,$$


### Example 2
- Input Image:
  <img src=./images/566cf0c6f5.png alt="\dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) } ; ~ ~ ~ \dot { z } _ { 2 } = - \frac { z _ { 2 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) }"  style="display:block; margin:auto;" />

- Target Latex:
  ```
  \dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) } ; ~ ~ ~ \dot { z } _ { 2 } = - \frac { z _ { 2 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) }
  ```

- Inferred Latex:
  ```
  \dot { z } _ { 1 } = - N ^ { 4 } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { F _ { 2 } ( z _ { 2 } - z _ { 1 } ) } , \quad \dot { z } _ { 2 } = - \frac { z _ { 2 } } { F _ { 2 } ( z _ { 2 } - z _ { 1 } ) }
  ```

- Render Inferred Latex:
  $$\dot { z } _ { 1 } = - N ^ { 4 } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { F _ { 2 } ( z _ { 2 } - z _ { 1 } ) } , \quad \dot { z } _ { 2 } = - \frac { z _ { 2 } } { F _ { 2 } ( z _ { 2 } - z _ { 1 } ) }$$

### Example 3
- Input Image:
  <img src=./images/4c0185889d.png alt="\dot { z } _ { 1 } = - N ^ { z } ( z _ { 1 } ) = - g ( z _ { 1 } ) = - \frac { z _ { 1 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) } ; ~ ~ ~ \dot { z } _ { 2 } = - \frac { z _ { 2 } } { P _ { z } ( z _ { 2 } - z _ { 1 } ) }"  style="display:block; margin:auto;" />

- Target Latex:
  ```
  { \cal L } ( J ) = \frac { 1 } { 2 } \partial _ { \mu } \phi \partial ^ { \mu } \phi + \frac { J } { 2 } \phi ^ { 2 } + \frac { \lambda \mu ^ { 2 \varepsilon } } { 4 ! } \phi ^ { 4 } + { \cal L } _ { \mathrm { C T } } ( J ) - \mu ^ { - 2 \varepsilon } \frac { \zeta } { 2 } \; J ^ { 2 } .
  ```

- Inferred Latex:
  ```
  { \cal L } ( J ) = \frac { 1 } { 2 } \partial _ { \mu } \phi \partial ^ { \mu } \phi + \frac { J } { 2 } \phi ^ { 2 } + \frac { \lambda \mu ^ { 2 x } } { 4 ! } \phi ^ { 4 } + { \cal L } _ { C \Gamma } ( J ) - \mu ^ { - 2 } \frac { \zeta } { 2 } } f ^ { 4 } ( x ) - \mu ^ { - 2 } \frac { \zeta } { 2 } f ( x ) = ( t ) .
  ```

- Render Inferred Latex:
  
  We get Latex that cannot be parsed because of the extra "}"
  $${ \cal L } ( J ) = \frac { 1 } { 2 } \partial _ { \mu } \phi \partial ^ { \mu } \phi + \frac { J } { 2 } \phi ^ { 2 } + \frac { \lambda \mu ^ { 2 x } } { 4 ! } \phi ^ { 4 } + { \cal L } _ { C \Gamma } ( J ) - \mu ^ { - 2 } \frac { \zeta } { 2 } } f ^ { 4 } ( x ) - \mu ^ { - 2 } \frac { \zeta } { 2 } f ( x ) = ( t ) .$$
  
  If we manually delete "}", we will get the following result:
  $${ \cal L } ( J ) = \frac { 1 } { 2 } \partial _ { \mu } \phi \partial ^ { \mu } \phi + \frac { J } { 2 } \phi ^ { 2 } + \frac { \lambda \mu ^ { 2 x } } { 4 ! } \phi ^ { 4 } + { \cal L } _ { C \Gamma } ( J ) - \mu ^ { - 2 } \frac { \zeta } { 2 }  f ^ { 4 } ( x ) - \mu ^ { - 2 } \frac { \zeta } { 2 } f ( x ) = ( t ) .$$


## Conclusion
Maybe we can add the restriction to force parentheses must be paired.


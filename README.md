# wenyang_pan_nlp_project_2021

## Project Charter 
This project aims to achieve two goals:
+ Develop NLP models to classify whether a comment is a toxic comment 
+ Have a mechanism to identify and document potential bias for these models

You can read more about this proejct in our [paper](https://github.com/MSIA/wenyang_pan_nlp_project_2021/blob/main/papers/NLP_Final_Report.pdf).

## Web App :rocket:
To get a quick feeling about what the models in this project can and can't accomplish, you can visit [this app](https://share.streamlit.io/msia/wenyang_pan_nlp_project_2021/main/app.py) and play with it!

## Setup

### Python Dependencies
We use [poetry](https://python-poetry.org/) to manage dependencies in this project.

You first need to install poetry with the instructions [here](https://python-poetry.org/docs/master/).

Now you can install all dependencies with the following.
```
poetry install
```

Now you can activate the virtual environment with the following.
```
poetry shell
```

You can exit the poetry shell by typing `exit` in the command line.

### Kaggle Authentication 
If you want to use command line to interact with Kaggle like we describe below, you need to set up the API token with instructions [here](https://www.kaggle.com/docs/api#authentication).

### HuggingFace Hub :hugs:
We store one of our model (DistilBERT) in HuggingFace Hub to allow our app to download and use the model. You can create a huggingface account [here] and authenticate in your local machine with the followign command.
```
huggingface-cli login
```

## Data Pipeline

### Raw Data
The dataset comes from this [Kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). 

You can download the dataset with the following command given you finish the kaggle authentication described in the setup section above. 

```zsh
make raw-get
```

### Processed Train and Test Data
There are few commmon processing we apply to the training data and test data. We finish those steps with the following commands.
```zsh
make train-prepare
make test-prepare
```
The output of those commands are stored in `data/interim/train.parquet` and `data/interim/test.parquet`.

### Processed FastText Data
The `fastText` library requires a specical data format. You can get this format by running the following command.
```zsh
make fasttext-prepare
```

### Others
The rest of data processing steps are directly done in jupyter notebooks under the `notebooks/` folder. We will describe them in the next section.

## Model Development 
We train three different models for this project. This section describes how to get the trained model.

### Logistic Regression
The code for training the logistic regression model with bag of words encoding can be found in `notebooks/01_bag_of_word.ipynb` and running the notebook will save the preprocessor and model in `models/bow/`.

### FastText
The code for training the fastText model can be found in `notebooks/02_fasttext.ipynb`. The trained model will be saved in `models/ft.bin`.

### DistilBERT
The training for DistilBERT model requires more computational power. We first preprocess the data in `notebooks/03_transformers_data.ipynb` and running code in this notebook will save the preprocessed data in `data/interim/toxic_comments/`. 

The actual training for the model will require a GPU. We will Kaggle for this project but you can choose whatever platform you like. The code for training the model is located in `04_transformers_model.ipynb`.

`05_transformers_eval_explain.ipynb` shows our code to evaluate and explain the model, given that you download your trained model from the GPU platform.

`06_share_transformer_models.ipynb` show how to push the model in huggingface hub, given you finish the huggingface hub section above. You can learn more the hub [here](https://huggingface.co/course/chapter4/1?fw=pt). You can see our trained model in the hub from [this link](https://huggingface.co/martin-ha/toxic-comment-model).

## App Development 
We develop the web by using the [streamlit library](https://docs.streamlit.io/). The code for the app is in `app.py`. You can launch the web locally running the following command.
```
streamlit run app.py
```
Alternatively, as a reminder, we have a deployed version [here](https://share.streamlit.io/msia/wenyang_pan_nlp_project_2021/main/app.py).
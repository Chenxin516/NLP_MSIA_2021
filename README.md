# wenyang_pan_nlp_project_2021
Project for Text Analytics Class

## Setup
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

## Data Pipeline

### Raw Data
The dataset comes from this [Kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). 

You can download the dataset with the following command. Note that you need to first set up the API token with instructions [here](https://www.kaggle.com/docs/api#authentication).

```zsh
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
```

Now you can move it to data folder and unzip it.
```zsh
mv jigsaw-unintended-bias-in-toxicity-classification.zip data/
unzip data/jigsaw-unintended-bias-in-toxicity-classification.zip -d data/raw/
```

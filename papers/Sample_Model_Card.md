## Model description
This model is a fine-tuned version of the [DistilBERT model](https://huggingface.co/transformers/model_doc/distilbert.html) to classify toxic comments. 

## How to use

You can use the model with the following code.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.config.id2label = {0: "normal", 1: "toxic"}
model.config.label2id = {"normal": 0, "toxic": 1}

pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)
print(pipeline('This is a test text.'))
```

## Limitations and Bias

This model is intended to use for classify toxic online classifications. However, one limitation of the model is that it performs poorly for some comments that mention a specific identity subgroup, like Muslim. The following table shows a evaluation score for different identity group. You can learn the specific meaning of this metrics [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation). But basically, those metrics shows how well a model performs for a specific group. The larger the number, the better.

| **subgroup**                  | **subgroup_size** | **subgroup_auc**   | **bpsn_auc**       | **bnsp_auc**       |
| ----------------------------- | ----------------- | ------------------ | ------------------ | ------------------ |
| muslim                        | 108               | 0.6889880952380950 | 0.8112554112554110 | 0.8800518892055840 |
| jewish                        | 40                | 0.7489177489177490 | 0.8595356359015830 | 0.8250611265982460 |
| homosexual_gay_or_lesbian     | 56                | 0.7953125          | 0.7061053984575840 | 0.9722888937377260 |
| black                         | 84                | 0.8658307210031350 | 0.7582412358882950 | 0.9754200596128560 |
| white                         | 112               | 0.87578125         | 0.7843339895013120 | 0.9701402586017970 |
| female                        | 306               | 0.8982545764154960 | 0.886766007294528  | 0.9482218495745610 |
| christian                     | 231               | 0.9040551839464880 | 0.9168973860121720 | 0.9300520888699900 |
| male                          | 225               | 0.9216823785351700 | 0.8621754516176060 | 0.967060717060717  |
| psychiatric_or_mental_illness | 26                | 0.9236111111111110 | 0.9067005937234950 | 0.9500707444492820 |

The table above shows that the model performs poorly for the muslim and jewish group. In fact, you pass the sentence "Muslims are people who follow or practice Islam, an Abrahamic monotheistic religion." Into the model, the model will classify it as toxic. Be mindful for this type of potential bias.

## Training data
The training data comes this [Kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). We use 10% of the `train.csv` data to train the model.

## Training procedure

You can see [this documentation and codes](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) for how we train the model. It takes about 3 hours in a P-100 GPU.

## Evaluation results

The model achieves 94% accuracy and 0.59 f1-score in a 10000 rows held-out test set.
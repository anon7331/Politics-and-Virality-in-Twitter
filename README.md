# Politic-and-Virality-in-Twitter
In this repo we provide some of the resources used in our 'Politics and Virality in the Time of Twitter: A Large-Scale Cross-Party
Sentiment Analysis in Greece, Spain and United Kingdom' paper. In specific:
- the code used for the training of our models (./code/inetune_models.py and ./code/finetune_multi_cv.py)
- the model that was selected and used for the sentiment analysis
- the manually annotated data used for training are shared (./data/annotation/).
- the ids of tweets that were used in our analyis and control experiments (./data/main/ & ./data/control)
- a Jupyter Notebook containing the major parts of our analysis.

## Annotated Data (./data/annotation/)
- One folder for each language (English, Spanish, Greek).
- In each directory there are three files:
    1. *_900.csv  contains the 900 tweets that annotators labelled individually (300 tweets each annotator).
    2. *_tiebreak_100.csv contains the initial 100 tweets all annotators labelled. 'annotator_3' indicates the annotator that was used as a tiebreaker.
    3. *_combined.csv contains all tweets labelled for the language.


## Model
While we plan to upload all the models trained for our experiments to huggingface.co, currently only the main model used in our analysis can be currently be find at: https://anonymshare.com/jNYr/xlm-roberta-sentiment-multilingual.zip.

The model, 'xlm-roberta-sentiment-multilingual', is based on the implementation of 'cardiffnlp/twitter-xlm-roberta-base-sentiment' while being further finetuned on the annotated dataset.

### Example usage
```
from transformers import AutoModelForSequenceClassification, pipeline
model = AutoModelForSequenceClassification.from_pretrained('./xlm-roberta-sentiment-multilingual/')
sentiment_analysis_task = pipeline("sentiment-analysis", model=model, tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

sentiment_analysis_task('Today is a good day')
Out: [{'label': 'Positive', 'score': 0.978614866733551}]
```

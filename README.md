# Politics-in-the-Time-of-Twitter
In this repo we provide some of the resources used in our 'Politics in the Time of Twitter: A Large-Scale Cross-partySentiment Analysis in Greece, Spain and United Kingdom' paper. In specific our model used for sentiment analysis and the manually annotated data used for training are shared.

## Annotated Data (./data/annotation/)
- One folder for each language (English, Spanish, Greek).
- In each directory there are three files:
    1. *_900.csv  contains the 900 tweets that annotators labelled individually (300 tweets each annotator).
    2. *_tiebreak_100.csv contains the initial 100 tweets all annotators labelled. 'annotator_3' indicates the annotator that was used as a tiebreaker.
    3. *_combined.csv contains all tweets labelled for the language.


## Model
While we plan to upload all the models trained for our experiments to huggingface.co, currently only the main model used in our analysis can be currently be find at: https://anonymshare.com/d/r11E/xlm-roberta-sentiment-multilingual.

The model, 'xlm-roberta-sentiment-multilingual', is based on the implementation of 'cardiffnlp/twitter-xlm-roberta-base-sentiment' while being further finetuned on the annotated dataset.

### Example usage
```
from transformers import AutoModelForSequenceClassification, pipeline
model = AutoModelForSequenceClassification.from_pretrained('./xlm-roberta-sentiment-multilingual/')
sentiment_analysis_task = pipeline("sentiment-analysis", model=model, tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

sentiment_analysis_task('Today is a good day')
Out: [{'label': 'Positive', 'score': 0.978614866733551}]
```

# Deploy BERT for Sentiment Analsysis with FastAPI

Deploy a pre-trained BERT model for Sentiment Analysis as a REST API using FastAPI

## Demo

The model is trained to classify sentiment (negative, neutral, and positive) on a custom dataset from app reviews on Google Play. Here's a sample request to the API:

```bash
 curl -d "{\"text\":\"This game is amazing, it is literally part of my childhood. It works well with hand eye coordination, and might even help with reflexes (not positive, just a guess)This game can keep you interested for hours,and has a lot of small things to work for! I really like the way the game has been moving as of update.\"}" -X POST http://localhost:8000/predict
```

The response you'll get looks something like this:

```js
{
    "probabilities": {
        "negative": 2.0558945834636688e-05,
        "neutral": 4.625277506420389e-05,
        "positive": 0.9999332427978516
    },
    "sentiment": "positive",
    "confidence": 0.9999332427978516
}
```

## Installation

Clone this repo:

```sh
git clone https://github.com/kandeldeepak46/Fine-Tuning-BERT-For-Sentiment-Analysis-Served-With-fastAPI.git
cd Fine-Tuning-BERT-For-Sentiment-Analysis-Served-With-fastAPI
```

Install the dependencies:

```sh
pipenv install --dev
```

Download the pre-trained model:

```sh
bin/download_model
```

## Test the setup

Start the HTTP server:

```sh
bin/start_server
```

Send a test request:

```sh
bin/test_request
```

## License

MIT

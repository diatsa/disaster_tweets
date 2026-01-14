# Disaster Tweets Classification
A machine learning project to classify tweets as either describing real disasters or not, using natural language processing and transformer-based models.

## Team Members
Diana (diatsa), Cindy (jellothebirdie), Linda (Quinn1108)

## About the Project
This is a one-day project developed for the Upskill Software Development Lifecycle program. The goal is to build a model that can automatically identify tweets about real disasters versus tweets using disaster-related words figuratively.
# Disaster Tweets Classification
A machine learning project to classify tweets as either describing real disasters or not, using natural language processing and transformer-based models.

## Team Members
Diana (diatsa), Cindy (jellothebirdie), Linda (Quinn1108)

## About the Project
This is a one-day project developed for the Upskill Software Development Lifecycle program. The goal is to build a model that can automatically identify tweets about real disasters versus tweets using disaster-related words figuratively.

## Problem Statement
Twitter has become a crucial communication channel during emergencies, with people using smartphones to report disasters in real-time. Disaster relief organizations and news agencies need to programmatically monitor Twitter, but it's not always clear whether a tweet is announcing an actual disaster or using disaster-related language metaphorically.

Challenge: Build a machine learning model that predicts which tweets are about real disasters and which ones aren't.

## Source
Kaggle NLP Getting Started Competition: https://www.kaggle.com/competitions/nlp-getting-started/overview

Starter code: 
* https://www.kaggle.com/code/philculliton/nlp-getting-started-tutorial/notebook
* https://www.kaggle.com/code/alexia/kerasnlp-starter-notebook-disaster-tweets

## Labels
Binary classification (disaster/not disaster)

## Approach
We used a transformer-based approach leveraging pre-trained language models:
* Data Exploration: Analyzed tweet patterns, word distributions, and class balance
* Data Preprocessing: Cleaned and prepared text data for model input
* Model Selection: Implemented DistilBERT from Keras NLP
* Fine-tuning: Trained and fine-tuned BERT for disaster tweet classification
* Evaluation: Generated predictions for submission

## Technologies Used
* Python
* Keras / TensorFlow
* Keras NLP
* DistilBERT / BERT
* Natural Language Processing (NLP)

## Repository Structure (to be updated)
```text
disaster_tweets/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for exploration and modeling
├── models/             # Saved model files
├── submissions/        # Kaggle submission files
└── README.md
```

## Acknowledgments
* Kaggle for providing the competition and dataset
* Upskill Software Development Lifecycle program
* The NLP and machine learning community for pre-trained models and tools

## License
This project was created for educational purposes as part of the Upskill program.
## Problem Statement
Twitter has become a crucial communication channel during emergencies, with people using smartphones to report disasters in real-time. Disaster relief organizations and news agencies need to programmatically monitor Twitter, but it's not always clear whether a tweet is announcing an actual disaster or using disaster-related language metaphorically.

Challenge: Build a machine learning model that predicts which tweets are about real disasters and which ones aren't.

## Source
Kaggle NLP Getting Started Competition: https://www.kaggle.com/competitions/nlp-getting-started/overview

Starter code: 
* https://www.kaggle.com/code/philculliton/nlp-getting-started-tutorial/notebook
* https://www.kaggle.com/code/alexia/kerasnlp-starter-notebook-disaster-tweets

## Labels
Binary classification (disaster/not disaster)

## Approach
We used a transformer-based approach leveraging pre-trained language models:
* Data Exploration: Analyzed tweet patterns, word distributions, and class balance
* Data Preprocessing: Cleaned and prepared text data for model input
* Model Selection: Implemented DistilBERT from Keras NLP
* Fine-tuning: Trained and fine-tuned BERT for disaster tweet classification
* Evaluation: Generated predictions for submission

## Technologies Used
* Python
* Keras / TensorFlow
* Keras NLP
* DistilBERT / BERT
* Natural Language Processing (NLP)

## Repository Structure (to be updated)
disaster_tweets/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for exploration and modeling
├── models/             # Saved model files
├── submissions/        # Kaggle submission files
└── README.md

## Acknowledgments
* Kaggle for providing the competition and dataset
* Upskill Software Development Lifecycle program
* The NLP and machine learning community for pre-trained models and tools

## License
This project was created for educational purposes as part of the Upskill program.
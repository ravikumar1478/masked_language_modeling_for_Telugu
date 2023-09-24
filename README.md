---
license: mit
tags:
- generated_from_trainer
model-index:
- name: xlm-roberta-base-finetuned-wikitext2
  results: []
language:
- en
metrics:
- accuracy
- code_eval
pipeline_tag: fill-mask
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# xlm-roberta-base-finetuned-wikitext2

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 2.0384

## Model description

We developed a language model for Telugu using the dataset called Telugu_books, which is from the Kaggle platform, and the dataset contains Telugu data,
there are only a few language models developed for regional languages like Telugu, Hindi, Kannada...etc, 
so we built a dedicated language model, especially for the Telugu language.
The model aim is to predict a Telugu word that is masked in a given Telugu sentence by using Masked Language Modeling of BERT [Bidirectional Encoder Representation from Transformers]
and we achieved state-of-the-art performance in it.


## Intended uses & limitations

Using this model we can predict the exact and contextual word which is already masked in a given Telugu sentence and we achieved state-of-the-art performance in it.

## Training and evaluation data

Training data:
Required libraries like Trainer and Training arguments are imported from 
transformers library. The after giving the Training arguments with our data we 
train the model using train() method which takes 1 to 1 ½ hour depending upon 
the size of our input data

Evaluating data:
In the hugging face after opening our model page there is an API in which We 
give a Telugu Sentence as input with <mask> keyword and click the compute
button then the predicted words with their probabilities are displayed. Then we 
check that words with the actual words and evaluated

## Training procedure

#### Step-1: 
Collecting Data
From the Kaggle Telugu dataset is collected. It contains Telugu paragraphs from 
different books.

#### Step-2: 
Pre-processing Data
The collected data is pre-processed using different pre-processing techniques 
and splitting the large Telugu Sentence into small sentences.

#### Step-3: 
Connecting to Hugging Face
Hugging Face provides a token with which we can log in using a notebook 
function and the rest of the work we do will be exported to the platform 
automatically.

#### Step-4: 
Loading pre-trained model and tokenizer
The pre-trained model and tokenizer from xlm-roberta-base are loaded for 
training our Telugu data

#### Step-5: 
Training the model
Required libraries like Trainer and Training arguments are imported from 
Transformers library. The after giving the Training arguments with our data we 
train the model using the train() method which takes 1 to 1 ½ hours depending upon 
the size of our input data

#### Step-6: 
Pushing model and tokenizer 
Then trainer.push_to_hub() and tokenizer.push_to_hub() methods are used to 
export our trained model and its tokenizers which are used for the mapping of 
words in prediction. 

#### Step-7: 
Testing
In the hugging face after opening our model page there is an API in which We 
give a Telugu Sentence as input with <mask> keyword and click the compute
button then the predicted words with their probabilities are displayed. Then we 
check that words with the actual words and evaluated

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 2.4192        | 1.0   | 1250 | 2.1557          |
| 2.2859        | 2.0   | 2500 | 2.0632          |
| 2.2311        | 3.0   | 3750 | 2.0083          |


### Framework versions

- Transformers 4.24.0
- Pytorch 1.12.1+cu113
- Datasets 2.7.1
- Tokenizers 0.13.2

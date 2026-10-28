# Presentation script

BEN DRIF ABDELHAMID

The slide numbers match the final presentation. Timing is a rehearsal target: 8 minutes, including the demonstration.

## Slide 1: Introduction

Hello everyone. My name is Ben Drif Abdelhamid.

Today, I will present my project about movie review sentiment analysis.

The main goal is to predict whether a movie review is positive or negative.

For this project, I compared two machine learning approaches: SVM and CNN.

## Slide 2: Presentation Outline

Here is the plan of my presentation.

First, I will introduce the dataset and explain how I prepared the text.

Then, I will present the two models, compare their results, and discuss their limitations.

Finally, I will show you the application and conclude with possible improvements.

## Slide 3: Dataset and Experimental Split

Let's start with the dataset.

I used the IMDb dataset from Kaggle, which contains 50,000 English movie reviews.

Half of them are positive, and the other half are negative.

After removing 422 duplicates, I divided the data into three parts: 80% for training, 10% for validation, and 10% for testing.

We use training data to learn, validation data to select the model, and test data for the final evaluation.

## Slide 4: Preprocessing

Before training the models, I needed to clean the text.

First, I removed HTML tags and converted the text to lowercase.

Then, I handled negation. For example, "didn't" becomes "did not".

I also used tokenization to split sentences into words, and lemmatization to simplify words. For example, "movies" becomes "movie".

An important point is that I kept negation words like "not", because "good" and "not good" have different meanings.

## Slide 5: Text Representation

After cleaning the text, we need to convert it into numbers that the models can understand.

For the SVM, I used TF-IDF, which gives importance to words and pairs of words.

For the CNN, I used GloVe embeddings, which represent each word as a vector of 100 numbers.

The CNN also uses up to 20,000 vocabulary entries and a maximum of 450 tokens per review.

Both vocabularies were built using training reviews only.

## Slide 6: TF-IDF with SVM

The first model is Support Vector Machine, or SVM.

It learns to separate positive reviews from negative reviews using TF-IDF features.

I chose SVM because it is efficient and works well with text classification.

We can also understand some of its decisions by looking at word weights.

For example, "excellent" has a positive weight, while "awful" has a negative weight.

## Slide 7: GloVe with CNN

The second model is a Convolutional Neural Network, or CNN.

It uses GloVe embeddings to represent words.

Then, a convolutional layer with 128 filters looks for patterns in groups of five words.

Global max pooling keeps the most important patterns, and the final layers predict the sentiment.

I also used dropout and early stopping to help reduce overfitting.

## Slide 8: Test Results

Now, let's compare the results.

Both models were evaluated on the same test set of 4,958 reviews.

The SVM achieved 90.86% accuracy, while the CNN achieved 87.11%.

The SVM also performed better in F1-score and ROC-AUC.

So, based on these results, we can say that SVM performed better than CNN in this experiment.

## Slide 9: Confusion Matrices

To better understand these results, I analyzed the confusion matrices.

They show correct predictions and classification errors.

A false positive happens when a negative review is predicted as positive.

A false negative is the opposite.

As we can see, SVM made 453 errors, while CNN made 639 errors.

This confirms the better performance of SVM.

## Slide 10: Overfitting

Now, let's look at the CNN training curves.

We can see that training accuracy continues to increase, but validation performance stops improving.

This means the model is learning the training data better, without improving on new data.

This problem is called overfitting.

To reduce it, I used early stopping.

Training stopped after six epochs and restored epoch four, which had the lowest validation loss.

## Slide 11: Error Analysis and Limitations

However, both models have limitations.

For example, this review contains both positive and negative opinions.

The SVM predicts positive, while the CNN predicts negative.

This illustrates a challenge with mixed opinions. This example has no official label, so disagreement alone does not tell us which model is correct.

Also, this experiment focuses only on English movie reviews and two sentiment classes.

So, more testing is needed to evaluate generalization.

## Slide 12: Application Demonstration

Finally, I developed a local application to test both models.

The user enters an English movie review and clicks "Analyze Sentiment".

The application displays the predictions from SVM and CNN and shows whether they agree.

[Enter a review, click the button, and show the predictions.]

As you can see, we can easily compare both models using this interface.

## Slide 13: Conclusion and Future Work

To conclude, this project covers the complete NLP pipeline, from data preprocessing to model training and application development.

The main result is that TF-IDF with SVM performed better than GloVe with CNN in this experiment.

For future work, I would like to improve the CNN, test different configurations, and evaluate the models on other datasets.

## Slide 14: Thank you for your attention

Thank you for your attention.

I am ready for your questions.


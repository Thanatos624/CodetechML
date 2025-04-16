# ML Model Suite

A comprehensive collection of machine learning models implemented in Python for various tasks including classification, sentiment analysis, computer vision, and recommendation systems.

## Project Overview

This repository contains implementations of several machine learning models:

1. **Classification with Decision Trees**: Implementation of a decision tree classifier using the Iris dataset, with visualization of the decision boundaries.

2. **NLP Sentiment Analysis**: A text classification pipeline for sentiment analysis of product reviews, using TF-IDF vectorization and Logistic Regression.

3. **Computer Vision with CNNs**: A convolutional neural network for image classification using the CIFAR-10 dataset.

4. **Recommendation Systems**: 
   - Collaborative filtering model for music recommendations
   - Hybrid recommendation system combining collaborative filtering with content-based features
   - Performance comparison between both approaches

## Decision Tree Classification

The decision tree classifier is trained on the Iris dataset to predict flower species based on petal and sepal measurements. The model achieves good accuracy and the decision boundaries are visualized to show how the model makes predictions.

![image](https://github.com/user-attachments/assets/7e289ee2-6db5-43b7-99f4-33f6f68aad84)


*Decision tree showing the classification process for the Iris dataset. The tree uses features like petal length and width to distinguish between flower species.*

## Sentiment Analysis

The sentiment analysis pipeline processes product reviews and classifies them as positive, neutral, or negative. The implementation includes:

- Text preprocessing (tokenization, stemming, stopword removal)
- TF-IDF vectorization
- Logistic regression classification
- Handling of missing values and empty texts

## CNN Image Classification

A convolutional neural network implemented with TensorFlow/Keras for the CIFAR-10 dataset, which contains 10 classes of common objects. The model architecture includes:

- Multiple convolutional and pooling layers
- Dense layers for classification
- Training with categorical cross-entropy loss

![image](https://github.com/user-attachments/assets/88e132d9-c487-48fc-bf31-9568f927cac0)


*Training and validation accuracy curves for the CNN model. The model achieved a test accuracy of 70.61% on the CIFAR-10 dataset. The lower image shows a successful prediction of a cat image.*

## Recommendation Systems

Two recommendation approaches are implemented for a Spotify songs dataset:

### Collaborative Filtering
- Uses matrix factorization to learn user-item interactions
- Predicts user preferences based on similar users' behaviors

### Hybrid Model
- Combines collaborative filtering with song audio features
- Incorporates content-based information like danceability, energy, tempo

![image](https://github.com/user-attachments/assets/8e5bf9cc-001e-4e09-9943-572ac2798c61)


*Comparison of loss curves between Collaborative Filtering (left) and Hybrid Model (right). The hybrid model demonstrates significantly lower MSE, showing the advantage of combining collaborative filtering with content-based features.*

## Requirements

```
numpy
pandas
scikit-learn
tensorflow
matplotlib
nltk
```

## Performance Metrics

- Decision Tree: Classification accuracy, confusion matrix, and classification report
- Sentiment Analysis: Accuracy, precision, recall, and F1-score
- CNN: Training and validation accuracy curves
- Recommendation Systems: MSE loss curves for both models

## Future Work

- Hyperparameter tuning for all models
- Implementing more advanced architectures (Transformers for NLP, ResNet for vision)
- Adding user interface for model interaction
- Exploring ensemble methods to improve performance

## Directory Structure

```
├── models/
│   ├── decision_tree.py
│   ├── sentiment_analysis.py
│   ├── cnn_classifier.py
│   └── recommendation_systems.py
├── data/
│   ├── iris_data.csv
│   ├── product_reviews.csv
│   └── spotify_songs.csv
├── images/
│   ├── decision_tree.png
│   ├── cnn_training.png
│   └── recommendation_loss.png
├── notebooks/
│   ├── decision_tree_analysis.ipynb
│   ├── sentiment_analysis_exploration.ipynb
│   ├── cnn_training.ipynb
│   └── recommendation_systems_comparison.ipynb
└── README.md
```

## Usage

Each model can be run independently. Examples:

```python
# Decision Tree
python models/decision_tree.py

# Sentiment Analysis
python models/sentiment_analysis.py

# CNN Classifier
python models/cnn_classifier.py

# Recommendation Systems
python models/recommendation_systems.py
```

Refer to individual Python files for more detailed usage instructions and parameters.

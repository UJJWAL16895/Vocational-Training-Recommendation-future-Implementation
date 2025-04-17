# Models Directory

This directory contains all machine learning models and training scripts for the vocational training recommendation system.

## Approach

The recommendation system will use a hybrid approach combining:

1. **Collaborative Filtering**: Recommending training programs based on similar user profiles and choices
2. **Content-Based Filtering**: Matching user preferences and background with training program features
3. **Market-Aware Recommendations**: Incorporating regional employment data to prioritize in-demand skills

## Model Files

- `data_processor.py` - Prepares data for model training
- `model_training.py` - Scripts for training and evaluating models
- `recommendation_engine.py` - Core recommendation algorithm implementation
- `evaluation.py` - Functions to evaluate recommendation quality

## Model Selection

Potential models to explore:

- K-Nearest Neighbors for similarity matching
- Matrix Factorization for collaborative filtering
- Random Forest or Gradient Boosting for prediction of program success
- Neural networks for complex pattern recognition

## Evaluation Metrics

- Precision and Recall
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- User satisfaction metrics (to be collected after deployment)
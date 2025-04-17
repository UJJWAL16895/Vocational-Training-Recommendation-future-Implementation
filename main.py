import os
import sys
import argparse
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from data.preprocessing import DataPreprocessor
from models.model_training import ModelTrainer
from models.recommendation_engine import RecommendationEngine
from models.evaluation import RecommendationEvaluator

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Vocational Training Recommendation System')
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--train', action='store_true', help='Train recommendation models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate recommendation models')
    parser.add_argument('--run-app', action='store_true', help='Run the web application')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline')
    
    return parser.parse_args()

def preprocess_data():
    """
    Preprocess data for the recommendation system.
    """
    print("\n===== Data Preprocessing =====")
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    employment_df = preprocessor.preprocess_employment_data()
    programs_df = preprocessor.preprocess_training_programs()
    
    # Create feature vectors
    feature_vectors = preprocessor.create_feature_vectors(programs_df)
    
    print("Data preprocessing complete!")
    return preprocessor

def train_models():
    """
    Train recommendation models.
    """
    print("\n===== Model Training =====")
    trainer = ModelTrainer()
    
    # Train all models
    trainer.train_all_models()
    
    print("Model training complete!")
    return trainer

def evaluate_models():
    """
    Evaluate recommendation models.
    """
    print("\n===== Model Evaluation =====")
    
    # Create recommendation engine with trained models
    recommendation_engine = RecommendationEngine()
    
    # Create evaluator
    evaluator = RecommendationEvaluator(recommendation_engine)
    
    # Evaluate all metrics
    results = evaluator.evaluate_all(k=10)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot evaluation results
    print("\nGenerating evaluation plots...")
    evaluator.plot_evaluation_results()
    
    print("Model evaluation complete!")
    return evaluator

def run_app():
    """
    Run the web application.
    """
    print("\n===== Running Web Application =====")
    
    # Change to app directory
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
    os.chdir(app_dir)
    
    # Run the Flask app
    from .app import app
    from flask import Flask
    app = Flask(__name__)
    app.run(debug=True)

def main():
    """
    Main function to run the entire pipeline.
    """
    args = parse_args()
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the entire pipeline if --all is specified
    if args.all:
        args.preprocess = True
        args.train = True
        args.evaluate = True
        args.run_app = True
    
    # Run data preprocessing
    if args.preprocess:
        preprocessor = preprocess_data()
    
    # Train models
    if args.train:
        trainer = train_models()
    
    # Evaluate models
    if args.evaluate:
        evaluator = evaluate_models()
    
    # Run the web application
    if args.run_app:
        run_app()

if __name__ == "__main__":
    main()
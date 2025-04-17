import os
import sys
import time
from models.model_training import ModelTrainer

def train_models():
    """
    Train all recommendation models and save them to the models/trained directory.
    This function provides a user-friendly interface for the training process.
    """
    print("\n===== Vocational Training Recommendation System =====\n")
    print("Starting model training process...\n")
    
    # Create the trainer instance
    trainer = ModelTrainer()
    
    # Train content-based model
    print("\n[1/3] Training content-based filtering model...")
    start_time = time.time()
    trainer.train_content_model()
    content_time = time.time() - start_time
    print(f"Content-based model training completed in {content_time:.2f} seconds.")
    
    # Train collaborative filtering model
    print("\n[2/3] Training collaborative filtering model...")
    start_time = time.time()
    trainer.train_collaborative_model()
    collab_time = time.time() - start_time
    print(f"Collaborative filtering model training completed in {collab_time:.2f} seconds.")
    
    # Train market-aware model
    print("\n[3/3] Training market-aware recommendation model...")
    start_time = time.time()
    trainer.train_market_model()
    market_time = time.time() - start_time
    print(f"Market-aware model training completed in {market_time:.2f} seconds.")
    
    # Print summary
    total_time = content_time + collab_time + market_time
    print("\n===== Training Summary =====")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Models saved to: {os.path.abspath(trainer.models_dir)}")
    print("\nAll models have been successfully trained and saved!")
    print("You can now use the recommendation engine to get personalized recommendations.")

if __name__ == "__main__":
    train_models()
import os
import sys
import time
import pandas as pd
import traceback
from models.model_training import ModelTrainer

def fix_training_data():
    """
    Fix the training data to ensure proper format for model training.
    """
    print("Preparing training data...")
    
    # Fix the training programs data
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    training_file = os.path.join(raw_dir, 'training_programs.csv')
    
    if os.path.exists(training_file):
        df = pd.read_csv(training_file)
        
        # Convert skills from space-separated to comma-separated
        if 'skills' in df.columns:
            df['skills'] = df['skills'].apply(lambda x: ','.join(str(x).split()) if isinstance(x, str) else '')
            
        # Save the fixed data
        df.to_csv(training_file, index=False)
        print(f"✓ Fixed training data format in {training_file}")
    else:
        print(f"✗ Training data file not found: {training_file}")

def ensure_directories():
    """
    Ensure all required directories exist.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directories if they don't exist
    dirs = [
        os.path.join(base_dir, 'data', 'raw'),
        os.path.join(base_dir, 'data', 'processed'),
        os.path.join(base_dir, 'models', 'trained')
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")

def train_models():
    """
    Train all recommendation models and save them to the models/trained directory.
    This function provides a user-friendly interface for the training process.
    """
    print("\n" + "=" * 50)
    print("   VOCATIONAL TRAINING RECOMMENDATION SYSTEM")
    print("=" * 50 + "\n")
    
    # Ensure directories exist
    ensure_directories()
    
    # Fix the training data first
    fix_training_data()
    
    # Create the trainer instance with absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models', 'trained')
    
    print("\nInitializing model trainer...")
    trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
    
    try:
        # Train content-based model
        print("\n[1/3] Training content-based filtering model...")
        start_time = time.time()
        trainer.train_content_model()
        content_time = time.time() - start_time
        print(f"✓ Content-based model training completed in {content_time:.2f} seconds.")
        
        # Train collaborative filtering model
        print("\n[2/3] Training collaborative filtering model...")
        start_time = time.time()
        trainer.train_collaborative_model()
        collab_time = time.time() - start_time
        print(f"✓ Collaborative filtering model training completed in {collab_time:.2f} seconds.")
        
        # Train market-aware model
        print("\n[3/3] Training market-aware recommendation model...")
        start_time = time.time()
        trainer.train_market_model()
        market_time = time.time() - start_time
        print(f"✓ Market-aware model training completed in {market_time:.2f} seconds.")
        
        # Print summary
        total_time = content_time + collab_time + market_time
        print("\n" + "=" * 50)
        print("   TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Models saved to: {os.path.abspath(models_dir)}")
        print("\nAll models have been successfully trained and saved!")
        print("You can now use the recommendation engine to get personalized recommendations.")
        
        return True
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        traceback.print_exc()
        return False

def main():
    try:
        success = train_models()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
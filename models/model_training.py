import os
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the data preprocessor
from data.preprocessing import DataPreprocessor

class ModelTrainer:
    """
    Class for training machine learning models for the vocational training recommendation system.
    
    This class implements training for:
    1. Content-based filtering model
    2. Collaborative filtering model
    3. Market-aware recommendation model
    """
    
    def __init__(self, data_dir='../data', models_dir='../models/trained'):
        """
        Initialize the model trainer.
        
        Args:
            data_dir (str): Directory containing the data files
            models_dir (str): Directory to save trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.preprocessor = DataPreprocessor(data_dir)
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.content_model = None
        self.collaborative_model = None
        self.market_model = None
        
        # Import the API client for real-time data
        from data.api_client import EmploymentDataClient
        self.data_client = EmploymentDataClient()
        
        # Load or create data
        self.programs_df = self.preprocessor.load_training_programs()
        
        # Use real-time employment data instead of synthetic data
        try:
            print("Fetching real-time employment data for model training...")
            self.employment_df = self.data_client.fetch_employment_data(refresh_cache=True)
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            print("Falling back to sample employment data")
            self.employment_df = self.preprocessor.load_employment_data()
        
        # Preprocess data
        self.programs_df = self.preprocessor.preprocess_training_programs(self.programs_df)
        self.employment_df = self.preprocessor.preprocess_employment_data(self.employment_df)
        
        # Create feature vectors
        self.program_features = self.preprocessor.create_feature_vectors(self.programs_df)
    
    def train_content_model(self):
        """
        Train content-based filtering model using program features.
        
        This model calculates similarity between programs based on their features.
        """
        print("Training content-based filtering model...")
        
        # Create feature matrix
        feature_keys = sorted(list(self.program_features[list(self.program_features.keys())[0]].keys()))
        program_ids = list(self.program_features.keys())
        
        # Create feature matrix
        feature_matrix = np.zeros((len(program_ids), len(feature_keys)))
        for i, program_id in enumerate(program_ids):
            for j, feature in enumerate(feature_keys):
                feature_matrix[i, j] = self.program_features[program_id][feature]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(feature_matrix)
        
        # Create nearest neighbors model
        self.content_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.content_model.fit(feature_matrix)
        
        # Save model and metadata
        model_path = os.path.join(self.models_dir, 'content_model.joblib')
        metadata_path = os.path.join(self.models_dir, 'content_model_metadata.json')
        
        joblib.dump(self.content_model, model_path)
        
        metadata = {
            'program_ids': program_ids,
            'feature_keys': feature_keys,
            'similarity_matrix': similarity_matrix.tolist()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Content-based model saved to {model_path}")
        return self.content_model
    
    def train_collaborative_model(self, n_components=10):
        """
        Train collaborative filtering model using simulated user-program interactions.
        
        In a real implementation, this would use actual user interaction data.
        For demonstration, we'll create synthetic user-program interactions.
        
        Args:
            n_components (int): Number of latent factors for matrix factorization
        """
        print("Training collaborative filtering model...")
        
        # Create synthetic user-program interaction matrix
        # In a real implementation, this would come from actual user data
        n_users = 100  # Synthetic users
        n_programs = len(self.programs_df)
        
        # Create synthetic user-program interaction matrix with some patterns
        interaction_matrix = np.zeros((n_users, n_programs))
        
        # Simulate user preferences with some patterns
        for user_id in range(n_users):
            # Assign user to a preference group (0-5)
            preference_group = user_id % 6
            
            # Set preferences based on group
            for prog_id in range(n_programs):
                program = self.programs_df.iloc[prog_id]
                
                # Get program features
                if 'id' in program and program['id'] in self.program_features:
                    features = self.program_features[program['id']]
                else:
                    continue
                
                # Set interaction based on preference group
                if preference_group == 0 and features['technology'] > 0.7:
                    interaction_matrix[user_id, prog_id] = np.random.uniform(0.7, 1.0)
                elif preference_group == 1 and features['healthcare'] > 0.7:
                    interaction_matrix[user_id, prog_id] = np.random.uniform(0.7, 1.0)
                elif preference_group == 2 and features['trades'] > 0.7:
                    interaction_matrix[user_id, prog_id] = np.random.uniform(0.7, 1.0)
                elif preference_group == 3 and features['business'] > 0.7:
                    interaction_matrix[user_id, prog_id] = np.random.uniform(0.7, 1.0)
                elif preference_group == 4 and features['creative'] > 0.7:
                    interaction_matrix[user_id, prog_id] = np.random.uniform(0.7, 1.0)
                elif preference_group == 5 and features['education'] > 0.7:
                    interaction_matrix[user_id, prog_id] = np.random.uniform(0.7, 1.0)
                else:
                    # Add some noise
                    if np.random.random() < 0.1:
                        interaction_matrix[user_id, prog_id] = np.random.uniform(0.1, 0.5)
        
        # Apply matrix factorization using TruncatedSVD
        svd = TruncatedSVD(n_components=n_components)
        latent_factors = svd.fit_transform(interaction_matrix)
        
        # Save the model
        self.collaborative_model = {
            'svd': svd,
            'latent_factors': latent_factors
        }
        
        # Save model and metadata
        model_path = os.path.join(self.models_dir, 'collaborative_model.joblib')
        metadata_path = os.path.join(self.models_dir, 'collaborative_model_metadata.json')
        
        joblib.dump(self.collaborative_model, model_path)
        
        metadata = {
            'n_users': n_users,
            'n_programs': n_programs,
            'n_components': n_components,
            'explained_variance_ratio': svd.explained_variance_ratio_.tolist(),
            'program_ids': self.programs_df['id'].tolist() if 'id' in self.programs_df.columns else list(range(n_programs))
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Collaborative filtering model saved to {model_path}")
        return self.collaborative_model
    
    def train_market_model(self):
        """
        Train market-aware recommendation model using regional employment data.
        
        This model clusters regions based on employment data and calculates
        program demand scores for each cluster.
        """
        print("Training market-aware recommendation model...")
        
        # Prepare regional employment data
        regions = self.employment_df['region'].unique()
        industries = ['technology', 'healthcare', 'trades', 'business', 'creative', 'education']
        
        # Create region-industry matrix
        region_industry_matrix = np.zeros((len(regions), len(industries)))
        
        for i, region in enumerate(regions):
            region_data = self.employment_df[self.employment_df['region'] == region]
            for j, industry in enumerate(industries):
                industry_data = region_data[region_data['industry'] == industry]
                if not industry_data.empty:
                    # Use demand score as the value
                    region_industry_matrix[i, j] = industry_data['demand_score'].values[0]
        
        # Cluster regions based on employment data
        n_clusters = min(4, len(regions))  # Adjust based on number of regions
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(region_industry_matrix)
        
        # Calculate program demand scores for each cluster
        cluster_program_scores = {}
        
        for cluster_id in range(n_clusters):
            # Get regions in this cluster
            cluster_regions = [region for i, region in enumerate(regions) if cluster_labels[i] == cluster_id]
            
            # Get average industry demand for this cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_industry_demand = np.mean(region_industry_matrix[cluster_indices], axis=0)
            
            # Calculate program scores based on industry demand
            program_scores = {}
            for program_id, features in self.program_features.items():
                score = 0
                total_weight = 0
                
                for i, industry in enumerate(industries):
                    if industry in features:
                        industry_weight = features[industry]
                        industry_demand = cluster_industry_demand[i]
                        score += industry_weight * industry_demand
                        total_weight += industry_weight
                
                if total_weight > 0:
                    program_scores[program_id] = score / total_weight
                else:
                    program_scores[program_id] = 0.5  # Default score
            
            cluster_program_scores[cluster_id] = program_scores
        
        # Create region to cluster mapping
        region_cluster_map = {region: cluster_labels[i] for i, region in enumerate(regions)}
        
        # Save the model
        self.market_model = {
            'kmeans': kmeans,
            'industries': industries,
            'cluster_program_scores': cluster_program_scores,
            'region_cluster_map': region_cluster_map
        }
        
        # Save model and metadata
        model_path = os.path.join(self.models_dir, 'market_model.joblib')
        metadata_path = os.path.join(self.models_dir, 'market_model_metadata.json')
        
        joblib.dump(self.market_model, model_path)
        
        # Convert NumPy arrays to lists for JSON serialization
        metadata = {
            'n_clusters': n_clusters,
            'industries': industries,
            'regions': regions.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'region_cluster_map': region_cluster_map,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Market-aware model saved to {model_path}")
        return self.market_model
    
    def train_all_models(self):
        """
        Train all recommendation models and save them.
        """
        self.train_content_model()
        self.train_collaborative_model()
        self.train_market_model()
        
        print("All models trained successfully!")


if __name__ == "__main__":
    # Train all models
    trainer = ModelTrainer()
    trainer.train_all_models()
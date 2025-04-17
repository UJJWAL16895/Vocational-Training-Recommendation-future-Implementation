import os
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class RecommendationEngine:
    """
    Recommendation engine for vocational training programs based on
    user preferences and regional employment data.
    
    This class implements a hybrid recommendation approach combining:
    1. Content-based filtering: Matching user preferences with program features
    2. Collaborative filtering: Finding similar user profiles
    3. Market-aware recommendations: Incorporating regional employment data
    """
    
    def __init__(self, models_dir='trained'):
        """
        Initialize the recommendation engine.
        Loads trained models if available, otherwise uses sample data.
        
        Args:
            models_dir (str): Directory containing trained models
        """
        # Set models directory path
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), models_dir)
        
        # Import the API client for real-time data
        from data.api_client import EmploymentDataClient
        self.data_client = EmploymentDataClient()
        
        # Load training programs and employment data
        self.training_programs = self._load_sample_programs()
        self.employment_data = self._load_real_time_employment_data()
        self.region_weights = self._load_region_weights()
        
        # Initialize models
        self.content_model = None
        self.collaborative_model = None
        self.market_model = None
        self.content_metadata = None
        self.collaborative_metadata = None
        self.market_metadata = None
        
        # Try to load trained models
        self.is_trained = self._load_trained_models()
        
        if not self.is_trained:
            print("Warning: Using sample data instead of trained models.")
            print("Run model_training.py to train models for better recommendations.")
    
    def _load_trained_models(self):
        """
        Load trained models from the models directory.
        
        Returns:
            bool: True if all models were loaded successfully, False otherwise
        """
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_dir):
                return False
            
            # Load content-based model
            content_model_path = os.path.join(self.models_dir, 'content_model.joblib')
            content_metadata_path = os.path.join(self.models_dir, 'content_model_metadata.json')
            
            if os.path.exists(content_model_path) and os.path.exists(content_metadata_path):
                self.content_model = joblib.load(content_model_path)
                with open(content_metadata_path, 'r') as f:
                    self.content_metadata = json.load(f)
            else:
                return False
            
            # Load collaborative filtering model
            collab_model_path = os.path.join(self.models_dir, 'collaborative_model.joblib')
            collab_metadata_path = os.path.join(self.models_dir, 'collaborative_model_metadata.json')
            
            if os.path.exists(collab_model_path) and os.path.exists(collab_metadata_path):
                self.collaborative_model = joblib.load(collab_model_path)
                with open(collab_metadata_path, 'r') as f:
                    self.collaborative_metadata = json.load(f)
            else:
                return False
            
            # Load market-aware model
            market_model_path = os.path.join(self.models_dir, 'market_model.joblib')
            market_metadata_path = os.path.join(self.models_dir, 'market_model_metadata.json')
            
            if os.path.exists(market_model_path) and os.path.exists(market_metadata_path):
                self.market_model = joblib.load(market_model_path)
                with open(market_metadata_path, 'r') as f:
                    self.market_metadata = json.load(f)
            else:
                return False
            
            return True
        except Exception as e:
            print(f"Error loading trained models: {e}")
            return False
    
    def _load_sample_programs(self):
        """
        Load sample training program data.
        In a real implementation, this would load from a database or CSV files.
        """
        return [
            {
                "id": 1,
                "name": "Web Development Bootcamp",
                "duration": "12 weeks",
                "skills": ["HTML", "CSS", "JavaScript", "React"],
                "job_placement_rate": 85,
                "cost": "$8,000",
                "demand_score": 0.89,
                "features": {
                    "technology": 0.9,
                    "healthcare": 0.1,
                    "trades": 0.2,
                    "business": 0.3,
                    "creative": 0.6,
                    "education": 0.2
                },
                "duration_months": 3
            },
            {
                "id": 2,
                "name": "Data Science Fundamentals",
                "duration": "16 weeks",
                "skills": ["Python", "SQL", "Machine Learning", "Statistics"],
                "job_placement_rate": 78,
                "cost": "$10,000",
                "demand_score": 0.92,
                "features": {
                    "technology": 0.9,
                    "healthcare": 0.3,
                    "trades": 0.1,
                    "business": 0.7,
                    "creative": 0.2,
                    "education": 0.4
                },
                "duration_months": 4
            },
            {
                "id": 3,
                "name": "Healthcare Administration",
                "duration": "8 weeks",
                "skills": ["Medical Billing", "Healthcare Regulations", "Patient Management"],
                "job_placement_rate": 82,
                "cost": "$5,500",
                "demand_score": 0.75,
                "features": {
                    "technology": 0.3,
                    "healthcare": 0.9,
                    "trades": 0.1,
                    "business": 0.7,
                    "creative": 0.1,
                    "education": 0.3
                },
                "duration_months": 2
            },
            {
                "id": 4,
                "name": "Cybersecurity Specialist",
                "duration": "14 weeks",
                "skills": ["Network Security", "Ethical Hacking", "Security Compliance"],
                "job_placement_rate": 88,
                "cost": "$9,200",
                "demand_score": 0.95,
                "features": {
                    "technology": 0.9,
                    "healthcare": 0.2,
                    "trades": 0.1,
                    "business": 0.5,
                    "creative": 0.1,
                    "education": 0.2
                },
                "duration_months": 3.5
            },
            {
                "id": 5,
                "name": "Digital Marketing",
                "duration": "10 weeks",
                "skills": ["SEO", "Social Media Marketing", "Content Creation", "Analytics"],
                "job_placement_rate": 75,
                "cost": "$6,800",
                "demand_score": 0.82,
                "features": {
                    "technology": 0.6,
                    "healthcare": 0.1,
                    "trades": 0.1,
                    "business": 0.8,
                    "creative": 0.8,
                    "education": 0.3
                },
                "duration_months": 2.5
            },
            {
                "id": 6,
                "name": "Electrical Technician",
                "duration": "20 weeks",
                "skills": ["Circuit Analysis", "Electrical Installation", "Troubleshooting", "Safety Protocols"],
                "job_placement_rate": 85,
                "cost": "$7,500",
                "demand_score": 0.88,
                "features": {
                    "technology": 0.5,
                    "healthcare": 0.1,
                    "trades": 0.9,
                    "business": 0.2,
                    "creative": 0.1,
                    "education": 0.2
                },
                "duration_months": 5
            },
            {
                "id": 7,
                "name": "Graphic Design Certificate",
                "duration": "12 weeks",
                "skills": ["Adobe Creative Suite", "Typography", "Layout Design", "Brand Identity"],
                "job_placement_rate": 72,
                "cost": "$6,200",
                "demand_score": 0.78,
                "features": {
                    "technology": 0.5,
                    "healthcare": 0.0,
                    "trades": 0.1,
                    "business": 0.4,
                    "creative": 0.9,
                    "education": 0.2
                },
                "duration_months": 3
            },
            {
                "id": 8,
                "name": "Medical Assistant Training",
                "duration": "16 weeks",
                "skills": ["Patient Care", "Medical Office Procedures", "Basic Clinical Skills", "Medical Terminology"],
                "job_placement_rate": 80,
                "cost": "$5,800",
                "demand_score": 0.85,
                "features": {
                    "technology": 0.2,
                    "healthcare": 0.9,
                    "trades": 0.1,
                    "business": 0.3,
                    "creative": 0.1,
                    "education": 0.4
                },
                "duration_months": 4
            }
        ]
    
    def _load_real_time_employment_data(self):
        """
        Load real-time employment data by region using the API client.
        This replaces the synthetic data with actual market data.
        
        Returns:
            dict: Dictionary containing employment data by region and industry
        """
        try:
            # Fetch employment data for all regions
            employment_df = self.data_client.fetch_employment_data(region='all')
            
            # Process the DataFrame into the required dictionary format
            employment_data = {}
            
            # Group by region and create nested dictionaries
            for region in employment_df['region'].unique():
                region_data = employment_df[employment_df['region'] == region]
                
                # Create industry dictionary for this region
                industry_data = {}
                for _, row in region_data.iterrows():
                    industry = row['industry']
                    demand_score = row['demand_score']
                    industry_data[industry] = demand_score
                
                # Add to employment data dictionary
                employment_data[region] = industry_data
            
            print(f"Loaded real-time employment data for {len(employment_data)} regions")
            return employment_data
            
        except Exception as e:
            print(f"Error loading real-time employment data: {e}")
            print("Falling back to sample employment data")
            # Fall back to sample data if API fails
            return self._load_sample_employment_data()
    
    def _load_sample_employment_data(self):
        """
        Load sample employment data by region.
        This is used as a fallback if the API client fails.
        """
        return {
            "northeast": {
                "technology": 0.85,
                "healthcare": 0.78,
                "trades": 0.65,
                "business": 0.82,
                "creative": 0.75,
                "education": 0.70
            },
            "midwest": {
                "technology": 0.72,
                "healthcare": 0.80,
                "trades": 0.85,
                "business": 0.75,
                "creative": 0.60,
                "education": 0.72
            },
            "south": {
                "technology": 0.78,
                "healthcare": 0.82,
                "trades": 0.80,
                "business": 0.78,
                "creative": 0.65,
                "education": 0.75
            },
            "west": {
                "technology": 0.92,
                "healthcare": 0.75,
                "trades": 0.70,
                "business": 0.80,
                "creative": 0.85,
                "education": 0.72
            },
            "northwest": {
                "technology": 0.88,
                "healthcare": 0.76,
                "trades": 0.75,
                "business": 0.72,
                "creative": 0.80,
                "education": 0.70
            },
            "southwest": {
                "technology": 0.80,
                "healthcare": 0.78,
                "trades": 0.82,
                "business": 0.75,
                "creative": 0.70,
                "education": 0.68
            },
            "southeast": {
                "technology": 0.75,
                "healthcare": 0.85,
                "trades": 0.78,
                "business": 0.80,
                "creative": 0.68,
                "education": 0.76
            }
        }
    
    def _load_region_weights(self):
        """
        Load weights for different regions based on real-time market data.
        These weights are used to adjust recommendations based on regional economic strength.
        
        Returns:
            dict: Dictionary mapping regions to weight values
        """
        try:
            # Fetch market trends data which includes regional comparison information
            market_trends = self.data_client.fetch_market_trends()
            
            # Extract regional comparison data
            if 'regional_comparison' in market_trends:
                regional_data = market_trends['regional_comparison']
                
                # Calculate weights based on average demand scores across industries
                weights = {}
                for region, industries in regional_data.items():
                    # Calculate average demand score for this region
                    avg_demand = sum(industries.values()) / len(industries) if industries else 0.5
                    
                    # Convert to weight (normalize around 1.0)
                    # Regions with higher demand get higher weights
                    weights[region] = 0.8 + (avg_demand * 0.5)  # Scale to range approximately 0.8-1.3
                
                print(f"Loaded real-time region weights for {len(weights)} regions")
                return weights
            else:
                print("No regional comparison data found in market trends")
                # Fall back to sample weights
                return self._load_sample_region_weights()
                
        except Exception as e:
            print(f"Error loading real-time region weights: {e}")
            print("Falling back to sample region weights")
            # Fall back to sample weights if API fails
            return self._load_sample_region_weights()
    
    def _load_sample_region_weights(self):
        """
        Load sample weights for different regions.
        This is used as a fallback if the API client fails.
        """
        return {
            "northeast": 1.2,
            "midwest": 1.1,
            "south": 1.0,
            "west": 1.3,
            "northwest": 1.2,
            "southwest": 1.0,
            "southeast": 1.1
        }
    
    def get_recommendations(self, user_data):
        """
        Generate personalized recommendations based on user data and regional employment trends.
        
        Args:
            user_data (dict): User preferences and background information
                - interests (list): List of interest areas
                - prior_experience (str): Level of prior work experience
                - education_level (str): Highest level of education
                - preferred_duration (str): Preferred program duration
                - region (str): User's region
        
        Returns:
            list: Recommended training programs sorted by relevance
        """
        # Extract user features
        user_interests = user_data.get('interests', [])
        user_region = user_data.get('region', 'northeast')  # Default to northeast if not specified
        preferred_duration = user_data.get('preferred_duration', 'medium')  # Default to medium if not specified
        prior_experience = user_data.get('prior_experience', 'none')  # Default to none if not specified
        education_level = user_data.get('education_level', 'high_school')  # Default to high school if not specified
        
        # Convert user interests to feature vector
        user_features = self._create_user_feature_vector(user_interests)
        
        # Get regional employment data
        regional_data = self.employment_data.get(user_region, self.employment_data['northeast'])
        region_weight = self.region_weights.get(user_region, 1.0)
        
        # Calculate program scores
        recommendations = []
        
        # If trained models are available, use them for better recommendations
        if self.is_trained and self.content_model and self.market_model:
            # Get content-based recommendations using trained model
            content_based_scores = self._get_content_based_scores(user_features)
            
            # Get collaborative filtering recommendations
            collaborative_scores = self._get_collaborative_scores(user_data)
            
            # Get market-aware recommendations
            market_scores = self._get_market_aware_scores(user_region)
            
            # Combine all scores for each program
            for program in self.training_programs:
                program_id = program['id']
                
                # Get scores from each model
                content_score = content_based_scores.get(program_id, 0.5)
                collab_score = collaborative_scores.get(program_id, 0.5)
                market_score = market_scores.get(program_id, 0.5)
                
                # Calculate duration preference score
                duration_score = self._calculate_duration_score(program, preferred_duration)
                
                # Calculate experience and education relevance
                exp_edu_score = self._calculate_experience_education_score(program, prior_experience, education_level)
                
                # Calculate final score (weighted combination)
                final_score = (
                    0.35 * content_score +  # Content-based filtering
                    0.20 * collab_score +   # Collaborative filtering
                    0.25 * market_score +   # Market-aware recommendations
                    0.10 * duration_score + # Duration preference
                    0.10 * exp_edu_score    # Experience and education relevance
                )
                
                # Create recommendation object
                recommendation = program.copy()
                recommendation['match_score'] = min(4, int(final_score * 5))  # Scale to 0-4 for UI display
                recommendation['content_score'] = content_score
                recommendation['collaborative_score'] = collab_score
                recommendation['market_score'] = market_score
                recommendation['duration_score'] = duration_score
                recommendation['exp_edu_score'] = exp_edu_score
                
                recommendations.append(recommendation)
        else:
            # Fallback to basic recommendation if models aren't available
            for program in self.training_programs:
                # Calculate content-based similarity score
                content_score = self._calculate_content_similarity(user_features, program['features'])
                
                # Calculate market demand score based on regional data
                market_score = self._calculate_market_score(program, regional_data, region_weight)
                
                # Calculate duration preference score
                duration_score = self._calculate_duration_score(program, preferred_duration)
                
                # Calculate experience and education relevance
                exp_edu_score = self._calculate_experience_education_score(program, prior_experience, education_level)
                
                # Calculate final score (weighted combination)
                final_score = (
                    0.5 * content_score + 
                    0.3 * market_score + 
                    0.1 * duration_score + 
                    0.1 * exp_edu_score
                )
                
                # Create recommendation object
                recommendation = program.copy()
                recommendation['match_score'] = min(4, int(final_score * 5))  # Scale to 0-4 for UI display
                recommendation['content_score'] = content_score
                recommendation['market_score'] = market_score
                recommendation['duration_score'] = duration_score
                recommendation['exp_edu_score'] = exp_edu_score
                
                recommendations.append(recommendation)
        
        # Filter recommendations to prioritize programs matching user interests
        if user_interests:
            # Get the primary interest categories
            primary_interests = [interest.lower() for interest in user_interests]
            
            # Separate recommendations into matching and non-matching groups
            matching_recommendations = []
            other_recommendations = []
            
            for rec in recommendations:
                # Check if program features match any user interests
                program_categories = [cat for cat, val in rec.get('features', {}).items() if val > 0.7]
                program_categories = [cat.lower() for cat in program_categories]
                
                # If any primary interest matches a program category, prioritize it
                if any(interest in program_categories for interest in primary_interests):
                    matching_recommendations.append(rec)
                else:
                    other_recommendations.append(rec)
            
            # Sort each group by match score and demand score
            matching_recommendations.sort(key=lambda x: (x['match_score'], x['demand_score']), reverse=True)
            other_recommendations.sort(key=lambda x: (x['match_score'], x['demand_score']), reverse=True)
            
            # Combine the groups, with matching recommendations first
            recommendations = matching_recommendations + other_recommendations
        else:
            # If no interests specified, just sort by match score and demand score
            recommendations.sort(key=lambda x: (x['match_score'], x['demand_score']), reverse=True)
        
        # Limit to top recommendations
        recommendations = recommendations[:8]
        
        return recommendations
    
    def _get_content_based_scores(self, user_features):
        """
        Get content-based recommendation scores using trained model.
        
        Args:
            user_features (dict): User interest feature vector
            
        Returns:
            dict: Dictionary mapping program IDs to content-based scores
        """
        # Convert user features to vector format expected by the model
        if self.content_metadata and 'feature_keys' in self.content_metadata:
            feature_keys = self.content_metadata['feature_keys']
            user_vector = np.array([user_features.get(feature, 0.0) for feature in feature_keys]).reshape(1, -1)
            
            # Get nearest neighbors
            # Check if content model exists before calling kneighbors
            if self.content_model is not None:
                distances, indices = self.content_model.kneighbors(user_vector, n_neighbors=len(self.training_programs))
            else:
                # Fallback if model is None - return default distances and indices
                distances = np.ones((1, len(self.training_programs)))
                indices = np.arange(len(self.training_programs)).reshape(1, -1)
            
            # Convert distances to similarity scores (1 - normalized distance)
            similarities = 1 - (distances[0] / np.max(distances[0]) if np.max(distances[0]) > 0 else distances[0])
            
            # Map program IDs to scores
            program_ids = self.content_metadata['program_ids']
            scores = {program_ids[idx]: sim for idx, sim in zip(indices[0], similarities)}
            
            return scores
        else:
            # Fallback to basic content similarity if metadata is not available
            return {program['id']: self._calculate_content_similarity(user_features, program['features']) 
                    for program in self.training_programs}
    
    def _get_collaborative_scores(self, user_data):
        """
        Get collaborative filtering recommendation scores.
        
        Args:
            user_data (dict): User preferences and background information
            
        Returns:
            dict: Dictionary mapping program IDs to collaborative filtering scores
        """
        if not self.collaborative_model or not self.collaborative_metadata:
            # Return default scores if model is not available
            return {program['id']: 0.5 for program in self.training_programs}
        
        # Create a feature vector for the user based on their preferences
        user_interests = user_data.get('interests', [])
        user_region = user_data.get('region', 'northeast')
        preferred_duration = user_data.get('preferred_duration', 'medium')
        
        # Map user to latent space
        # This is a simplified approach - in a real system, you would have actual user-item interactions
        svd = self.collaborative_model['svd']
        latent_factors = self.collaborative_model['latent_factors']
        
        # Create a pseudo-interaction vector for the user
        n_programs = len(self.training_programs)
        user_vector = np.zeros(n_programs)
        
        # Set higher values for programs matching user interests
        for i, program in enumerate(self.training_programs):
            match_score = 0.0
            
            # Match interests
            for interest in user_interests:
                if interest in program['features'] and program['features'][interest] > 0.7:
                    match_score += 0.3
            
            # Match region demand
            if 'demand_score' in program and program['demand_score'] > 0.8:
                match_score += 0.2
            
            # Match duration preference
            if self._calculate_duration_score(program, preferred_duration) > 0.8:
                match_score += 0.1
            
            user_vector[i] = min(1.0, match_score)
        
        # Transform user vector to latent space
        user_latent = svd.transform(user_vector.reshape(1, -1))[0]
        
        # Calculate similarity with existing latent factors
        similarities = cosine_similarity([user_latent], latent_factors)[0]
        
        # Get top similar users
        top_similar_users = np.argsort(similarities)[-5:]  # Get top 5 similar users
        
        # Get program scores based on similar users' preferences
        program_scores = {}
        program_ids = self.collaborative_metadata.get('program_ids', list(range(n_programs)))
        
        for i, program_id in enumerate(program_ids):
            if i < len(self.training_programs):
                # Calculate weighted average of similar users' ratings for this program
                weighted_sum = 0.0
                total_weight = 0.0
                
                for user_idx in top_similar_users:
                    if user_idx < len(latent_factors):
                        weight = max(0, similarities[user_idx])  # Use similarity as weight
                        if weight > 0:
                            # Get this user's predicted rating for the program
                            user_latent_factor = latent_factors[user_idx]
                            program_latent_factor = svd.components_[:, i]
                            rating = np.dot(user_latent_factor, program_latent_factor)
                            
                            weighted_sum += weight * rating
                            total_weight += weight
                
                if total_weight > 0:
                    # Normalize score to 0-1 range
                    score = weighted_sum / total_weight
                    score = max(0, min(1, (score + 1) / 2))  # Transform from [-1,1] to [0,1]
                    program_scores[program_id] = score
                else:
                    program_scores[program_id] = 0.5
        
        return program_scores
    
    def _get_market_aware_scores(self, user_region):
        """
        Get market-aware recommendation scores based on regional employment data.
        
        Args:
            user_region (str): User's region
            
        Returns:
            dict: Dictionary mapping program IDs to market-aware scores
        """
        if not self.market_model or not self.market_metadata:
            # Fallback to basic market score calculation if model is not available
            regional_data = self.employment_data.get(user_region, self.employment_data['northeast'])
            region_weight = self.region_weights.get(user_region, 1.0)
            return {program['id']: self._calculate_market_score(program, regional_data, region_weight) 
                    for program in self.training_programs}
        
        # Get cluster for user's region
        region_cluster_map = self.market_model['region_cluster_map']
        cluster_program_scores = self.market_model['cluster_program_scores']
        
        # Default to first cluster if region not found
        cluster_id = region_cluster_map.get(user_region, 0)
        
        # Get program scores for this cluster
        if cluster_id in cluster_program_scores:
            return cluster_program_scores[cluster_id]
        else:
            # Fallback to basic market score calculation
            regional_data = self.employment_data.get(user_region, self.employment_data['northeast'])
            region_weight = self.region_weights.get(user_region, 1.0)
            return {program['id']: self._calculate_market_score(program, regional_data, region_weight) 
                    for program in self.training_programs}
    
    def _calculate_experience_education_score(self, program, prior_experience, education_level):
        """
        Calculate score based on how well the program matches user's experience and education.
        
        Args:
            program (dict): Training program data
            prior_experience (str): User's prior experience level
            education_level (str): User's education level
            
        Returns:
            float: Experience and education match score between 0 and 1
        """
        # Map experience levels to numeric values
        experience_values = {
            'none': 0,
            'entry': 1,
            'mid': 2,
            'experienced': 3
        }
        
        # Map education levels to numeric values
        education_values = {
            'high_school': 0,
            'some_college': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4
        }
        
        # Get user's numeric values
        user_exp_value = experience_values.get(prior_experience, 0)
        user_edu_value = education_values.get(education_level, 0)
        
        # Calculate experience match score
        # Programs with higher job placement rates are better for less experienced users
        job_placement_rate = program.get('job_placement_rate', 75) / 100.0
        
        if user_exp_value <= 1:  # None or entry-level
            # Higher job placement is better for inexperienced users
            exp_score = job_placement_rate
        else:  # Mid or experienced
            # Experienced users care less about job placement
            exp_score = 0.5 + (job_placement_rate / 2)
        
        # Calculate education match score
        # This is a simplified approach - in a real system, you would have program-specific education requirements
        if 'technology' in program['features'] and program['features']['technology'] > 0.7:
            # Tech programs often require more education
            edu_score = min(1.0, (user_edu_value + 1) / 4)
        elif 'healthcare' in program['features'] and program['features']['healthcare'] > 0.7:
            # Healthcare programs often require specific education
            edu_score = min(1.0, (user_edu_value + 1) / 3)
        elif 'trades' in program['features'] and program['features']['trades'] > 0.7:
            # Trades programs often require less formal education
            edu_score = 0.7 + (user_edu_value / 10)
        else:
            # Default education score
            edu_score = 0.5 + (user_edu_value / 8)
        
        # Combine scores (weighted average)
        return (0.6 * exp_score) + (0.4 * edu_score)
    
    def _create_user_feature_vector(self, user_interests):
        """
        Convert user interests to a feature vector.
        
        Args:
            user_interests (list): List of user interest areas
            
        Returns:
            dict: Feature vector representing user interests
        """
        # Initialize feature vector with zeros
        features = {
            "technology": 0.0,
            "healthcare": 0.0,
            "trades": 0.0,
            "business": 0.0,
            "creative": 0.0,
            "education": 0.0
        }
        
        # Set features based on user interests
        for interest in user_interests:
            if interest.lower() == 'technology':
                features['technology'] = 1.0
            elif interest.lower() == 'healthcare':
                features['healthcare'] = 1.0
            elif interest.lower() == 'trades':
                features['trades'] = 1.0
            elif interest.lower() == 'business':
                features['business'] = 1.0
            elif interest.lower() == 'creative':
                features['creative'] = 1.0
            elif interest.lower() == 'education':
                features['education'] = 1.0
        
        return features
    
    def _calculate_content_similarity(self, user_features, program_features):
        """
        Calculate content-based similarity between user interests and program features.
        
        Args:
            user_features (dict): User interest feature vector
            program_features (dict): Program feature vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Convert dictionaries to vectors
        user_vector = np.array([user_features[feature] for feature in sorted(user_features.keys())])
        program_vector = np.array([program_features[feature] for feature in sorted(program_features.keys())])
        
        # If user has no interests selected, return a moderate score
        if np.sum(user_vector) == 0:
            return 0.5
        
        # Check for exact category match - strongly prioritize programs that match user's primary interests
        user_interests = [feature for feature, value in user_features.items() if value > 0.5]
        program_primary_categories = [feature for feature, value in program_features.items() if value > 0.7]
        
        # If there's a direct match between user interests and program's primary category, boost the score
        category_match_boost = 0.0
        for interest in user_interests:
            if interest in program_primary_categories:
                category_match_boost = 0.5  # Significant boost for exact category match
                break
        
        # Calculate cosine similarity
        dot_product = np.dot(user_vector, program_vector)
        user_norm = np.linalg.norm(user_vector)
        program_norm = np.linalg.norm(program_vector)
        
        if user_norm == 0 or program_norm == 0:
            return 0.0
        
        # Base similarity score
        similarity = dot_product / (user_norm * program_norm)
        
        # Apply category match boost (capped at 1.0)
        final_similarity = min(1.0, similarity + category_match_boost)
        return final_similarity
    
    def _calculate_market_score(self, program, regional_data, region_weight):
        """
        Calculate market demand score based on regional employment data.
        
        Args:
            program (dict): Training program data
            regional_data (dict): Regional employment data
            region_weight (float): Weight for the region
            
        Returns:
            float: Market demand score between 0 and 1
        """
        # Calculate weighted average of regional demand for program features
        total_weight = 0.0
        weighted_sum = 0.0
        
        for feature, program_value in program['features'].items():
            if program_value > 0:
                regional_value = regional_data.get(feature, 0.5)
                weighted_sum += program_value * regional_value
                total_weight += program_value
        
        if total_weight == 0:
            return 0.5
        
        # Apply region weight
        market_score = (weighted_sum / total_weight) * region_weight
        
        # Normalize to 0-1 range
        return min(1.0, market_score / 1.5)
    
    def _calculate_duration_score(self, program, preferred_duration):
        """
        Calculate score based on how well the program duration matches user preference.
        
        Args:
            program (dict): Training program data
            preferred_duration (str): User's preferred duration ('short', 'medium', 'long')
            
        Returns:
            float: Duration match score between 0 and 1
        """
        duration_months = program.get('duration_months', 3)  # Default to 3 months if not specified
        
        if preferred_duration == 'short':  # 1-3 months
            if duration_months <= 3:
                return 1.0
            elif duration_months <= 4:
                return 0.7
            else:
                return 0.4
        elif preferred_duration == 'medium':  # 3-6 months
            if 3 <= duration_months <= 6:
                return 1.0
            elif duration_months < 3 or duration_months <= 7:
                return 0.7
            else:
                return 0.4
        elif preferred_duration == 'long':  # 6+ months
            if duration_months >= 6:
                return 1.0
            elif duration_months >= 4:
                return 0.7
            else:
                return 0.4
        else:
            return 0.5  # No preference specified
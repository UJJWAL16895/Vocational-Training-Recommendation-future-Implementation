import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    Class for preprocessing regional employment data and vocational training program information
    for the recommendation system.
    """
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir (str): Directory containing the data files
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_employment_data(self, filename='employment_data.csv'):
        """
        Load regional employment data from CSV file.
        
        Args:
            filename (str): Name of the CSV file containing employment data
            
        Returns:
            pd.DataFrame: DataFrame containing employment data
        """
        file_path = os.path.join(self.raw_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded employment data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Using sample data instead")
            return self._create_sample_employment_data()
    
    def load_training_programs(self, filename='training_programs.csv'):
        """
        Load vocational training program data from CSV file.
        
        Args:
            filename (str): Name of the CSV file containing training program data
            
        Returns:
            pd.DataFrame: DataFrame containing training program data
        """
        file_path = os.path.join(self.raw_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded training program data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Using sample data instead")
            return self._create_sample_training_programs()
    
    def preprocess_employment_data(self, df=None):
        """
        Preprocess regional employment data.
        
        Args:
            df (pd.DataFrame): DataFrame containing employment data, or None to load from file
            
        Returns:
            pd.DataFrame: Preprocessed employment data
        """
        if df is None:
            df = self.load_employment_data()
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Normalize numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Save preprocessed data
        output_path = os.path.join(self.processed_dir, 'employment_data_processed.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved preprocessed employment data to {output_path}")
        
        return df
    
    def preprocess_training_programs(self, df=None):
        """
        Preprocess vocational training program data.
        
        Args:
            df (pd.DataFrame): DataFrame containing training program data, or None to load from file
            
        Returns:
            pd.DataFrame: Preprocessed training program data
        """
        if df is None:
            df = self.load_training_programs()
        
        # Handle missing values
        df = df.fillna('')
        
        # Extract skills as a list
        if 'skills' in df.columns and isinstance(df['skills'].iloc[0], str):
            df['skills_list'] = df['skills'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        
        # Convert duration to numeric (months)
        if 'duration' in df.columns:
            df['duration_months'] = df['duration'].apply(self._parse_duration)
        
        # Save preprocessed data
        output_path = os.path.join(self.processed_dir, 'training_programs_processed.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved preprocessed training program data to {output_path}")
        
        return df
    
    def _parse_duration(self, duration_str):
        """
        Parse duration string to numeric months.
        
        Args:
            duration_str (str): Duration string (e.g., '12 weeks', '3 months')
            
        Returns:
            float: Duration in months
        """
        if not isinstance(duration_str, str):
            return 3.0  # Default to 3 months
        
        parts = duration_str.lower().split()
        if len(parts) < 2:
            return 3.0
        
        try:
            value = float(parts[0])
            unit = parts[1]
            
            if 'week' in unit:
                return value / 4.0  # Convert weeks to months
            elif 'month' in unit:
                return value
            elif 'year' in unit:
                return value * 12.0  # Convert years to months
            else:
                return 3.0  # Default
        except (ValueError, IndexError):
            return 3.0  # Default
    
    def create_feature_vectors(self, programs_df):
        """
        Create feature vectors for training programs.
        
        Args:
            programs_df (pd.DataFrame): DataFrame containing training program data
            
        Returns:
            dict: Dictionary mapping program IDs to feature vectors
        """
        feature_vectors = {}
        
        for _, row in programs_df.iterrows():
            program_id = row['id']
            
            # Initialize feature vector with zeros
            features = {
                "technology": 0.0,
                "healthcare": 0.0,
                "trades": 0.0,
                "business": 0.0,
                "creative": 0.0,
                "education": 0.0
            }
            
            # Set features based on program category or skills
            if 'category' in row and row['category']:
                category = row['category'].lower()
                if 'tech' in category or 'software' in category or 'web' in category:
                    features['technology'] = 0.9
                elif 'health' in category or 'medical' in category:
                    features['healthcare'] = 0.9
                elif 'trade' in category or 'construction' in category or 'electrical' in category:
                    features['trades'] = 0.9
                elif 'business' in category or 'marketing' in category or 'finance' in category:
                    features['business'] = 0.9
                elif 'creative' in category or 'design' in category or 'art' in category:
                    features['creative'] = 0.9
                elif 'education' in category or 'teaching' in category:
                    features['education'] = 0.9
            
            # Adjust features based on skills
            if 'skills_list' in row and row['skills_list']:
                skills = row['skills_list']
                for skill in skills:
                    skill = skill.lower().strip()
                    if any(tech in skill for tech in ['programming', 'software', 'web', 'data', 'computer', 'it']):
                        features['technology'] = max(features['technology'], 0.8)
                    elif any(health in skill for health in ['medical', 'health', 'patient', 'clinical']):
                        features['healthcare'] = max(features['healthcare'], 0.8)
                    elif any(trade in skill for trade in ['electrical', 'plumbing', 'construction', 'mechanical']):
                        features['trades'] = max(features['trades'], 0.8)
                    elif any(business in skill for business in ['business', 'management', 'marketing', 'finance']):
                        features['business'] = max(features['business'], 0.8)
                    elif any(creative in skill for creative in ['design', 'creative', 'art', 'media']):
                        features['creative'] = max(features['creative'], 0.8)
                    elif any(education in skill for education in ['teaching', 'education', 'training']):
                        features['education'] = max(features['education'], 0.8)
            
            feature_vectors[program_id] = features
        
        # Save feature vectors
        output_path = os.path.join(self.processed_dir, 'program_features.json')
        with open(output_path, 'w') as f:
            json.dump(feature_vectors, f, indent=2)
        print(f"Saved program feature vectors to {output_path}")
        
        return feature_vectors
    
    def _create_sample_employment_data(self):
        """
        Create sample employment data for demonstration purposes.
        
        Returns:
            pd.DataFrame: Sample employment data
        """
        regions = ['northeast', 'midwest', 'south', 'west', 'northwest', 'southwest', 'southeast']
        industries = ['technology', 'healthcare', 'trades', 'business', 'creative', 'education']
        
        data = []
        for region in regions:
            for industry in industries:
                # Generate random employment metrics
                employment_rate = np.random.uniform(0.7, 0.95)
                job_growth = np.random.uniform(-0.02, 0.15)
                median_salary = np.random.uniform(35000, 120000)
                demand_score = np.random.uniform(0.5, 1.0)
                
                data.append({
                    'region': region,
                    'industry': industry,
                    'employment_rate': employment_rate,
                    'job_growth': job_growth,
                    'median_salary': median_salary,
                    'demand_score': demand_score
                })
        
        df = pd.DataFrame(data)
        
        # Save sample data
        output_path = os.path.join(self.raw_dir, 'employment_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Created and saved sample employment data to {output_path}")
        
        return df
    
    def _create_sample_training_programs(self):
        """
        Create sample training program data for demonstration purposes.
        
        Returns:
            pd.DataFrame: Sample training program data
        """
        programs = [
            {
                'id': 1,
                'name': 'Web Development Bootcamp',
                'category': 'Technology',
                'duration': '12 weeks',
                'skills': 'HTML,CSS,JavaScript,React',
                'job_placement_rate': 85,
                'cost': 8000,
                'description': 'Intensive bootcamp covering front-end and back-end web development.'
            },
            {
                'id': 2,
                'name': 'Data Science Fundamentals',
                'category': 'Technology',
                'duration': '16 weeks',
                'skills': 'Python,SQL,Machine Learning,Statistics',
                'job_placement_rate': 78,
                'cost': 10000,
                'description': 'Comprehensive program covering data analysis and machine learning.'
            },
            {
                'id': 3,
                'name': 'Healthcare Administration',
                'category': 'Healthcare',
                'duration': '8 weeks',
                'skills': 'Medical Billing,Healthcare Regulations,Patient Management',
                'job_placement_rate': 82,
                'cost': 5500,
                'description': 'Training in healthcare administration and medical office management.'
            },
            {
                'id': 4,
                'name': 'Cybersecurity Specialist',
                'category': 'Technology',
                'duration': '14 weeks',
                'skills': 'Network Security,Ethical Hacking,Security Compliance',
                'job_placement_rate': 88,
                'cost': 9200,
                'description': 'Training in cybersecurity principles and practices.'
            },
            {
                'id': 5,
                'name': 'Digital Marketing',
                'category': 'Business',
                'duration': '10 weeks',
                'skills': 'SEO,Social Media Marketing,Content Creation,Analytics',
                'job_placement_rate': 75,
                'cost': 6800,
                'description': 'Comprehensive digital marketing training program.'
            },
            {
                'id': 6,
                'name': 'Electrical Technician',
                'category': 'Trades',
                'duration': '20 weeks',
                'skills': 'Circuit Analysis,Electrical Installation,Troubleshooting,Safety Protocols',
                'job_placement_rate': 85,
                'cost': 7500,
                'description': 'Hands-on training for electrical installation and maintenance.'
            },
            {
                'id': 7,
                'name': 'Graphic Design Certificate',
                'category': 'Creative',
                'duration': '12 weeks',
                'skills': 'Adobe Creative Suite,Typography,Layout Design,Brand Identity',
                'job_placement_rate': 72,
                'cost': 6200,
                'description': 'Training in graphic design principles and tools.'
            },
            {
                'id': 8,
                'name': 'Medical Assistant Training',
                'category': 'Healthcare',
                'duration': '16 weeks',
                'skills': 'Patient Care,Medical Office Procedures,Basic Clinical Skills,Medical Terminology',
                'job_placement_rate': 80,
                'cost': 5800,
                'description': 'Comprehensive training for medical assistant roles.'
            }
        ]
        
        df = pd.DataFrame(programs)
        
        # Save sample data
        output_path = os.path.join(self.raw_dir, 'training_programs.csv')
        df.to_csv(output_path, index=False)
        print(f"Created and saved sample training program data to {output_path}")
        
        return df


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Create and preprocess sample data
    employment_df = preprocessor.preprocess_employment_data()
    programs_df = preprocessor.preprocess_training_programs()
    
    # Create feature vectors
    feature_vectors = preprocessor.create_feature_vectors(programs_df)
    
    print("Data preprocessing complete!")
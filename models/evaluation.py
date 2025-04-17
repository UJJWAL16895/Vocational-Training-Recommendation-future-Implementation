import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the recommendation engine and data preprocessor
from models.recommendation_engine import RecommendationEngine
from data.preprocessing import DataPreprocessor

class RecommendationEvaluator:
    """
    Class for evaluating the performance of the recommendation engine.
    
    This class implements various evaluation metrics for recommendation systems:
    1. Precision and Recall
    2. Mean Average Precision (MAP)
    3. Normalized Discounted Cumulative Gain (NDCG)
    4. Coverage and Diversity
    """
    
    def __init__(self, recommendation_engine=None):
        """
        Initialize the evaluator.
        
        Args:
            recommendation_engine (RecommendationEngine): Recommendation engine to evaluate
        """
        if recommendation_engine is None:
            self.recommendation_engine = RecommendationEngine()
        else:
            self.recommendation_engine = recommendation_engine
        
        # Create synthetic test data
        self.test_users = self._create_test_users()
        self.ground_truth = self._create_ground_truth()
    
    def _create_test_users(self, n_users=50):
        """
        Create synthetic test users with various preferences.
        
        Args:
            n_users (int): Number of test users to create
            
        Returns:
            list: List of user data dictionaries
        """
        # Define possible values for each field
        interests_options = [
            ['technology'], ['healthcare'], ['trades'], ['business'], ['creative'], ['education'],
            ['technology', 'business'], ['healthcare', 'education'], ['trades', 'technology'],
            ['business', 'creative'], ['creative', 'education'], ['technology', 'healthcare', 'business']
        ]
        
        prior_experience_options = ['none', 'entry', 'mid', 'experienced']
        education_level_options = ['high_school', 'some_college', 'associate', 'bachelor', 'master']
        preferred_duration_options = ['short', 'medium', 'long']
        region_options = ['northeast', 'midwest', 'south', 'west', 'northwest', 'southwest', 'southeast']
        
        # Create test users
        test_users = []
        for i in range(n_users):
            user = {
                'user_id': i,
                'interests': np.random.choice(interests_options),
                'prior_experience': np.random.choice(prior_experience_options),
                'education_level': np.random.choice(education_level_options),
                'preferred_duration': np.random.choice(preferred_duration_options),
                'region': np.random.choice(region_options)
            }
            test_users.append(user)
        
        return test_users
    
    def _create_ground_truth(self):
        """
        Create synthetic ground truth data for evaluation.
        
        In a real system, this would be based on actual user interactions.
        For demonstration, we'll create synthetic ground truth based on user preferences.
        
        Returns:
            dict: Dictionary mapping user IDs to lists of relevant program IDs
        """
        ground_truth = {}
        
        for user in self.test_users:
            user_id = user['user_id']
            
            # Get recommendations for this user
            recommendations = self.recommendation_engine.get_recommendations(user)
            
            # Select top programs as ground truth (with some randomness)
            top_programs = recommendations[:5]  # Top 5 recommendations
            
            # Add some randomness - remove some top programs and add some lower-ranked ones
            if len(top_programs) > 2 and np.random.random() < 0.3:
                # Remove 1-2 top programs
                n_remove = np.random.randint(1, min(3, len(top_programs)))
                for _ in range(n_remove):
                    idx = np.random.randint(0, len(top_programs))
                    top_programs.pop(idx)
            
            # Add some lower-ranked programs
            if len(recommendations) > 5 and np.random.random() < 0.3:
                lower_programs = recommendations[5:10]  # Programs ranked 6-10
                n_add = np.random.randint(1, min(3, len(lower_programs)))
                for _ in range(n_add):
                    idx = np.random.randint(0, len(lower_programs))
                    top_programs.append(lower_programs[idx])
            
            # Store program IDs as ground truth
            ground_truth[user_id] = [program['id'] for program in top_programs]
        
        return ground_truth
    
    def evaluate_precision_recall(self, k=5):
        """
        Evaluate precision and recall at k.
        
        Args:
            k (int): Number of top recommendations to consider
            
        Returns:
            tuple: (precision@k, recall@k)
        """
        precision_sum = 0.0
        recall_sum = 0.0
        
        for user in self.test_users:
            user_id = user['user_id']
            
            # Get recommendations for this user
            recommendations = self.recommendation_engine.get_recommendations(user)
            recommended_ids = [rec['id'] for rec in recommendations[:k]]
            
            # Get ground truth for this user
            relevant_ids = self.ground_truth.get(user_id, [])
            
            # Calculate precision and recall
            n_relevant_recommended = len(set(recommended_ids) & set(relevant_ids))
            
            precision = n_relevant_recommended / len(recommended_ids) if recommended_ids else 0
            recall = n_relevant_recommended / len(relevant_ids) if relevant_ids else 0
            
            precision_sum += precision
            recall_sum += recall
        
        # Calculate average precision and recall
        avg_precision = precision_sum / len(self.test_users) if self.test_users else 0
        avg_recall = recall_sum / len(self.test_users) if self.test_users else 0
        
        return avg_precision, avg_recall
    
    def evaluate_map(self, k=10):
        """
        Evaluate Mean Average Precision (MAP).
        
        Args:
            k (int): Number of top recommendations to consider
            
        Returns:
            float: MAP score
        """
        ap_sum = 0.0
        
        for user in self.test_users:
            user_id = user['user_id']
            
            # Get recommendations for this user
            recommendations = self.recommendation_engine.get_recommendations(user)
            recommended_ids = [rec['id'] for rec in recommendations[:k]]
            
            # Get ground truth for this user
            relevant_ids = self.ground_truth.get(user_id, [])
            
            if not relevant_ids:
                continue
            
            # Calculate average precision for this user
            hits = 0
            sum_precisions = 0.0
            
            for i, rec_id in enumerate(recommended_ids):
                if rec_id in relevant_ids:
                    hits += 1
                    precision_at_i = hits / (i + 1)
                    sum_precisions += precision_at_i
            
            ap = sum_precisions / len(relevant_ids) if relevant_ids else 0
            ap_sum += ap
        
        # Calculate MAP
        map_score = ap_sum / len(self.test_users) if self.test_users else 0
        
        return map_score
    
    def evaluate_ndcg(self, k=10):
        """
        Evaluate Normalized Discounted Cumulative Gain (NDCG).
        
        Args:
            k (int): Number of top recommendations to consider
            
        Returns:
            float: NDCG score
        """
        ndcg_sum = 0.0
        
        for user in self.test_users:
            user_id = user['user_id']
            
            # Get recommendations for this user
            recommendations = self.recommendation_engine.get_recommendations(user)
            recommended_ids = [rec['id'] for rec in recommendations[:k]]
            
            # Get ground truth for this user
            relevant_ids = self.ground_truth.get(user_id, [])
            
            if not relevant_ids:
                continue
            
            # Calculate DCG
            dcg = 0.0
            for i, rec_id in enumerate(recommended_ids):
                if rec_id in relevant_ids:
                    # Use binary relevance (1 if relevant, 0 if not)
                    # Position is i+1 (1-indexed)
                    dcg += 1.0 / np.log2(i + 2)  # log2(2) = 1, log2(3) = 1.585, etc.
            
            # Calculate ideal DCG (IDCG)
            idcg = 0.0
            for i in range(min(len(relevant_ids), k)):
                idcg += 1.0 / np.log2(i + 2)
            
            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcg_sum += ndcg
        
        # Calculate average NDCG
        avg_ndcg = ndcg_sum / len(self.test_users) if self.test_users else 0
        
        return avg_ndcg
    
    def evaluate_coverage(self, k=10):
        """
        Evaluate catalog coverage of recommendations.
        
        Args:
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Coverage score (percentage of programs recommended to at least one user)
        """
        # Get all program IDs
        all_program_ids = set(program['id'] for program in self.recommendation_engine.training_programs)
        
        # Get recommended program IDs for all users
        recommended_program_ids = set()
        
        for user in self.test_users:
            # Get recommendations for this user
            recommendations = self.recommendation_engine.get_recommendations(user)
            recommended_ids = [rec['id'] for rec in recommendations[:k]]
            
            # Add to set of all recommended programs
            recommended_program_ids.update(recommended_ids)
        
        # Calculate coverage
        coverage = len(recommended_program_ids) / len(all_program_ids) if all_program_ids else 0
        
        return coverage
    
    def evaluate_diversity(self, k=10):
        """
        Evaluate diversity of recommendations.
        
        Args:
            k (int): Number of top recommendations to consider
            
        Returns:
            float: Diversity score (average pairwise dissimilarity of recommendations)
        """
        diversity_sum = 0.0
        
        for user in self.test_users:
            # Get recommendations for this user
            recommendations = self.recommendation_engine.get_recommendations(user)[:k]
            
            if len(recommendations) < 2:
                continue
            
            # Calculate pairwise dissimilarity
            dissimilarity_sum = 0.0
            pair_count = 0
            
            for i in range(len(recommendations)):
                for j in range(i+1, len(recommendations)):
                    # Get feature vectors
                    prog_i_features = recommendations[i]['features']
                    prog_j_features = recommendations[j]['features']
                    
                    # Convert to vectors
                    feature_keys = sorted(prog_i_features.keys())
                    vec_i = np.array([prog_i_features[key] for key in feature_keys])
                    vec_j = np.array([prog_j_features[key] for key in feature_keys])
                    
                    # Calculate cosine similarity
                    similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                    
                    # Convert to dissimilarity (1 - similarity)
                    dissimilarity = 1.0 - similarity
                    
                    dissimilarity_sum += dissimilarity
                    pair_count += 1
            
            # Calculate average dissimilarity for this user
            if pair_count > 0:
                avg_dissimilarity = dissimilarity_sum / pair_count
                diversity_sum += avg_dissimilarity
        
        # Calculate average diversity across all users
        avg_diversity = diversity_sum / len(self.test_users) if self.test_users else 0
        
        return avg_diversity
    
    def evaluate_all(self, k=10):
        """
        Evaluate all metrics.
        
        Args:
            k (int): Number of top recommendations to consider
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        precision, recall = self.evaluate_precision_recall(k)
        map_score = self.evaluate_map(k)
        ndcg = self.evaluate_ndcg(k)
        coverage = self.evaluate_coverage(k)
        diversity = self.evaluate_diversity(k)
        
        return {
            'precision@k': precision,
            'recall@k': recall,
            'map@k': map_score,
            'ndcg@k': ndcg,
            'coverage': coverage,
            'diversity': diversity
        }
    
    def plot_evaluation_results(self):
        """
        Plot evaluation results for different values of k.
        """
        # Evaluate for different values of k
        k_values = [1, 3, 5, 10, 15, 20]
        precision_values = []
        recall_values = []
        map_values = []
        ndcg_values = []
        
        for k in k_values:
            precision, recall = self.evaluate_precision_recall(k)
            map_score = self.evaluate_map(k)
            ndcg = self.evaluate_ndcg(k)
            
            precision_values.append(precision)
            recall_values.append(recall)
            map_values.append(map_score)
            ndcg_values.append(ndcg)
        
        # Create figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot precision and recall
        axs[0, 0].plot(k_values, precision_values, 'o-', label='Precision@k')
        axs[0, 0].plot(k_values, recall_values, 's-', label='Recall@k')
        axs[0, 0].set_xlabel('k')
        axs[0, 0].set_ylabel('Score')
        axs[0, 0].set_title('Precision and Recall at k')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot MAP
        axs[0, 1].plot(k_values, map_values, 'o-', label='MAP@k')
        axs[0, 1].set_xlabel('k')
        axs[0, 1].set_ylabel('Score')
        axs[0, 1].set_title('Mean Average Precision at k')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot NDCG
        axs[1, 0].plot(k_values, ndcg_values, 'o-', label='NDCG@k')
        axs[1, 0].set_xlabel('k')
        axs[1, 0].set_ylabel('Score')
        axs[1, 0].set_title('Normalized Discounted Cumulative Gain at k')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Plot coverage and diversity
        coverage_values = [self.evaluate_coverage(k) for k in k_values]
        diversity_values = [self.evaluate_diversity(k) for k in k_values]
        
        axs[1, 1].plot(k_values, coverage_values, 'o-', label='Coverage')
        axs[1, 1].plot(k_values, diversity_values, 's-', label='Diversity')
        axs[1, 1].set_xlabel('k')
        axs[1, 1].set_ylabel('Score')
        axs[1, 1].set_title('Coverage and Diversity at k')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        # Adjust layout and save figure
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(results_dir, 'evaluation_results.png'))
        plt.close()
        
        # Create additional plots
        self._plot_user_interest_distribution()
        self._plot_recommendation_distribution()
    
    def _plot_user_interest_distribution(self):
        """
        Plot distribution of user interests in test data.
        """
        # Count interest frequencies
        interest_counts = {}
        for user in self.test_users:
            for interest in user['interests']:
                if interest in interest_counts:
                    interest_counts[interest] += 1
                else:
                    interest_counts[interest] = 1
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(interest_counts.keys(), interest_counts.values())
        plt.xlabel('Interest Area')
        plt.ylabel('Number of Users')
        plt.title('Distribution of User Interests')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        plt.savefig(os.path.join(results_dir, 'user_interest_distribution.png'))
        plt.close()
    
    def _plot_recommendation_distribution(self):
        """
        Plot distribution of recommended programs.
        """
        # Count program recommendation frequencies
        program_counts = {}
        
        for user in self.test_users:
            recommendations = self.recommendation_engine.get_recommendations(user)[:5]  # Top 5 recommendations
            for rec in recommendations:
                program_id = rec['id']
                program_name = rec['name']
                
                if program_name in program_counts:
                    program_counts[program_name] += 1
                else:
                    program_counts[program_name] = 1
        
        # Sort by frequency
        sorted_counts = dict(sorted(program_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(sorted_counts.keys(), sorted_counts.values())
        plt.xlabel('Program Name')
        plt.ylabel('Recommendation Frequency')
        plt.title('Distribution of Recommended Programs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        plt.savefig(os.path.join(results_dir, 'recommendation_distribution.png'))
        plt.close()


if __name__ == "__main__":
    # Example usage
    evaluator = RecommendationEvaluator()
    results = evaluator.evaluate_all(k=10)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    evaluator.plot_evaluation_results()
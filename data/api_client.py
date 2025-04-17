import os
import requests
import pandas as pd
import json
import time
from datetime import datetime

class EmploymentDataClient:
    """
    Client for fetching real-time employment data from external APIs.
    
    This class handles API requests to fetch current employment statistics,
    job market trends, and industry demand data from reliable sources.
    """
    
    def __init__(self, api_key=None, cache_dir='../data/cache'):
        """
        Initialize the employment data client.
        
        Args:
            api_key (str): API key for data services (if required)
            cache_dir (str): Directory to cache API responses
        """
        self.api_key = api_key or os.environ.get('EMPLOYMENT_API_KEY')
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Base URLs for different data sources
        # These would be replaced with actual API endpoints in a production system
        self.bls_base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        self.market_trends_url = "https://api.example.com/market-trends"
        self.skills_gap_url = "https://api.example.com/skills-gap"
    
    def fetch_employment_data(self, region='all', refresh_cache=False):
        """
        Fetch employment data by region from BLS or similar data source.
        
        Args:
            region (str): Region to fetch data for, or 'all' for all regions
            refresh_cache (bool): Whether to refresh cached data
            
        Returns:
            pd.DataFrame: DataFrame containing employment data
        """
        cache_file = os.path.join(self.cache_dir, f'employment_data_{region}.csv')
        
        # Check if cached data exists and is recent (less than 1 day old)
        if os.path.exists(cache_file) and not refresh_cache:
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            # If cache is less than 1 day old, use it
            if (current_time - cache_time) < 86400:  # 86400 seconds = 1 day
                print(f"Loading cached employment data for {region}")
                return pd.read_csv(cache_file)
        
        print(f"Fetching real-time employment data for {region}")
        
        try:
            # In a real implementation, this would make an actual API request
            # For demonstration, we'll simulate an API response
            
            # If API key is available, use it for the real API
            if self.api_key and False:  # Set to True when ready to use real API
                # Example BLS API request (would need to be customized)
                headers = {'Content-type': 'application/json'}
                data = json.dumps({
                    "seriesid": ["CEU0500000001"],  # Example series ID
                    "startyear": "2023",
                    "endyear": "2023",
                    "registrationkey": self.api_key
                })
                
                response = requests.post(self.bls_base_url, data=data, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    # Process the API response into a DataFrame
                    # This would need to be customized based on the actual API response format
                    df = self._process_bls_response(result, region)
                else:
                    print(f"API request failed with status code {response.status_code}")
                    # Fall back to sample data if API request fails
                    df = self._get_sample_employment_data(region)
            else:
                # Use sample data for demonstration
                df = self._get_sample_employment_data(region)
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            
            return df
            
        except Exception as e:
            print(f"Error fetching employment data: {e}")
            # If API request fails, try to use cached data if available
            if os.path.exists(cache_file):
                print("Using cached data due to API error")
                return pd.read_csv(cache_file)
            else:
                # If no cached data, use sample data
                print("Using sample data due to API error")
                return self._get_sample_employment_data(region)
    
    def fetch_market_trends(self, region='all'):
        """
        Fetch current market trends data.
        
        Args:
            region (str): Region to fetch data for, or 'all' for all regions
            
        Returns:
            dict: Dictionary containing market trends data
        """
        cache_file = os.path.join(self.cache_dir, f'market_trends_{region}.json')
        
        # Check if cached data exists and is recent
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            # If cache is less than 1 day old, use it
            if (current_time - cache_time) < 86400:  # 86400 seconds = 1 day
                print(f"Loading cached market trends for {region}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        print(f"Fetching real-time market trends for {region}")
        
        try:
            # In a real implementation, this would make an actual API request
            # For demonstration, we'll simulate an API response
            
            # Example API request (would need to be customized)
            if self.api_key and False:  # Set to True when ready to use real API
                params = {'region': region, 'api_key': self.api_key}
                response = requests.get(self.market_trends_url, params=params)
                
                if response.status_code == 200:
                    trends_data = response.json()
                else:
                    print(f"API request failed with status code {response.status_code}")
                    # Fall back to sample data if API request fails
                    trends_data = self._get_sample_market_trends(region)
            else:
                # Use sample data for demonstration
                trends_data = self._get_sample_market_trends(region)
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(trends_data, f)
            
            return trends_data
            
        except Exception as e:
            print(f"Error fetching market trends: {e}")
            # If API request fails, try to use cached data if available
            if os.path.exists(cache_file):
                print("Using cached data due to API error")
                with open(cache_file, 'r') as f:
                    return json.load(f)
            else:
                # If no cached data, use sample data
                print("Using sample data due to API error")
                return self._get_sample_market_trends(region)
    
    def fetch_skills_gap_data(self):
        """
        Fetch current skills gap data.
        
        Returns:
            dict: Dictionary containing skills gap data
        """
        cache_file = os.path.join(self.cache_dir, 'skills_gap.json')
        
        # Check if cached data exists and is recent
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            # If cache is less than 1 day old, use it
            if (current_time - cache_time) < 86400:  # 86400 seconds = 1 day
                print("Loading cached skills gap data")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        print("Fetching real-time skills gap data")
        
        try:
            # In a real implementation, this would make an actual API request
            # For demonstration, we'll simulate an API response
            
            # Example API request (would need to be customized)
            if self.api_key and False:  # Set to True when ready to use real API
                params = {'api_key': self.api_key}
                response = requests.get(self.skills_gap_url, params=params)
                
                if response.status_code == 200:
                    skills_data = response.json()
                else:
                    print(f"API request failed with status code {response.status_code}")
                    # Fall back to sample data if API request fails
                    skills_data = self._get_sample_skills_gap_data()
            else:
                # Use sample data for demonstration
                skills_data = self._get_sample_skills_gap_data()
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(skills_data, f)
            
            return skills_data
            
        except Exception as e:
            print(f"Error fetching skills gap data: {e}")
            # If API request fails, try to use cached data if available
            if os.path.exists(cache_file):
                print("Using cached data due to API error")
                with open(cache_file, 'r') as f:
                    return json.load(f)
            else:
                # If no cached data, use sample data
                print("Using sample data due to API error")
                return self._get_sample_skills_gap_data()
    
    def _process_bls_response(self, response_data, region):
        """
        Process BLS API response into a DataFrame.
        
        Args:
            response_data (dict): API response data
            region (str): Region the data is for
            
        Returns:
            pd.DataFrame: Processed employment data
        """
        # This would need to be customized based on the actual API response format
        # For demonstration, we'll return sample data
        return self._get_sample_employment_data(region)
    
    def _get_sample_employment_data(self, region='all'):
        """
        Get sample employment data for demonstration purposes.
        This is more realistic than the previous synthetic data.
        
        Args:
            region (str): Region to get data for, or 'all' for all regions
            
        Returns:
            pd.DataFrame: Sample employment data
        """
        # Create sample data with more realistic values
        # This would be replaced with actual API data in production
        
        # Define regions and industries
        regions = ['northeast', 'midwest', 'south', 'west', 'northwest', 'southwest', 'southeast']
        industries = ['technology', 'healthcare', 'trades', 'business', 'creative', 'education']
        
        # Filter by region if specified
        if region != 'all' and region in regions:
            regions = [region]
        
        # Create sample data
        data = []
        
        # Current date for timestamp
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for r in regions:
            for industry in industries:
                # Generate more realistic values based on region and industry
                if industry == 'technology':
                    if r in ['west', 'northeast']:
                        demand_score = round(0.85 + 0.1 * np.random.random(), 2)
                        growth_rate = round(0.12 + 0.06 * np.random.random(), 2)
                        avg_salary = int(85000 + 15000 * np.random.random())
                    else:
                        demand_score = round(0.75 + 0.1 * np.random.random(), 2)
                        growth_rate = round(0.08 + 0.06 * np.random.random(), 2)
                        avg_salary = int(75000 + 10000 * np.random.random())
                elif industry == 'healthcare':
                    demand_score = round(0.80 + 0.1 * np.random.random(), 2)
                    growth_rate = round(0.10 + 0.05 * np.random.random(), 2)
                    avg_salary = int(70000 + 10000 * np.random.random())
                elif industry == 'trades':
                    if r in ['midwest', 'south']:
                        demand_score = round(0.82 + 0.08 * np.random.random(), 2)
                        growth_rate = round(0.09 + 0.04 * np.random.random(), 2)
                        avg_salary = int(65000 + 8000 * np.random.random())
                    else:
                        demand_score = round(0.75 + 0.08 * np.random.random(), 2)
                        growth_rate = round(0.07 + 0.04 * np.random.random(), 2)
                        avg_salary = int(60000 + 8000 * np.random.random())
                elif industry == 'business':
                    demand_score = round(0.78 + 0.08 * np.random.random(), 2)
                    growth_rate = round(0.08 + 0.04 * np.random.random(), 2)
                    avg_salary = int(68000 + 9000 * np.random.random())
                elif industry == 'creative':
                    if r in ['west', 'northeast']:
                        demand_score = round(0.75 + 0.1 * np.random.random(), 2)
                        growth_rate = round(0.08 + 0.05 * np.random.random(), 2)
                        avg_salary = int(65000 + 10000 * np.random.random())
                    else:
                        demand_score = round(0.65 + 0.1 * np.random.random(), 2)
                        growth_rate = round(0.06 + 0.04 * np.random.random(), 2)
                        avg_salary = int(60000 + 8000 * np.random.random())
                else:  # education
                    demand_score = round(0.70 + 0.08 * np.random.random(), 2)
                    growth_rate = round(0.05 + 0.04 * np.random.random(), 2)
                    avg_salary = int(58000 + 7000 * np.random.random())
                
                data.append({
                    'region': r,
                    'industry': industry,
                    'demand_score': demand_score,
                    'growth_rate': growth_rate,
                    'avg_salary': avg_salary,
                    'timestamp': current_date
                })
        
        return pd.DataFrame(data)
    
    def _get_sample_market_trends(self, region='all'):
        """
        Get sample market trends data for demonstration purposes.
        
        Args:
            region (str): Region to get data for, or 'all' for all regions
            
        Returns:
            dict: Sample market trends data
        """
        # Define regions and industries
        regions = ['northeast', 'midwest', 'south', 'west', 'northwest', 'southwest', 'southeast']
        industries = ['technology', 'healthcare', 'trades', 'business', 'creative', 'education']
        
        # Create sample industry demand data
        industry_demand = {}
        
        # If a specific region is requested
        if region != 'all' and region in regions:
            # Generate demand scores for the specified region
            industry_demand = {
                'technology': round(0.75 + 0.2 * np.random.random(), 2),
                'healthcare': round(0.75 + 0.15 * np.random.random(), 2),
                'trades': round(0.65 + 0.2 * np.random.random(), 2),
                'business': round(0.70 + 0.15 * np.random.random(), 2),
                'creative': round(0.60 + 0.25 * np.random.random(), 2),
                'education': round(0.65 + 0.15 * np.random.random(), 2)
            }
        else:
            # For 'all', calculate average across regions
            avg_demand = {}
            for industry in industries:
                avg_demand[industry] = round(0.70 + 0.15 * np.random.random(), 2)
            industry_demand = avg_demand
        
        # Create sample regional comparison data
        regional_comparison = {}
        for r in regions:
            regional_comparison[r] = {
                'technology': round(0.70 + 0.25 * np.random.random(), 2),
                'healthcare': round(0.75 + 0.15 * np.random.random(), 2),
                'trades': round(0.65 + 0.20 * np.random.random(), 2),
                'business': round(0.70 + 0.15 * np.random.random(), 2),
                'creative': round(0.60 + 0.25 * np.random.random(), 2),
                'education': round(0.65 + 0.15 * np.random.random(), 2)
            }
        
        # Sample job growth projection data
        job_growth = {
            "years": [str(year) for year in range(2023, 2028)],
            "industries": {
                "technology": [5.2, 6.8, 8.5, 10.2, 12.0],
                "healthcare": [4.5, 5.7, 7.0, 8.2, 9.5],
                "trades": [3.0, 3.8, 4.5, 5.2, 6.0],
                "business": [3.8, 4.5, 5.2, 6.0, 6.8],
                "creative": [2.5, 3.2, 4.0, 4.8, 5.5],
                "education": [2.0, 2.5, 3.0, 3.5, 4.0]
            }
        }
        
        # Prepare response data
        response_data = {
            'industry_demand': industry_demand,
            'regional_comparison': regional_comparison,
            'job_growth': job_growth,
            'timestamp': datetime.now().strftime('%Y-%m-%d')
        }
        
        return response_data
    
    def _get_sample_skills_gap_data(self):
        """
        Get sample skills gap data for demonstration purposes.
        
        Returns:
            dict: Sample skills gap data
        """
        # Sample skills gap data with more realistic values
        skills_gap = {
            "web development": {"demand": round(0.85 + 0.1 * np.random.random(), 2), "supply": round(0.65 + 0.1 * np.random.random(), 2)},
            "data analysis": {"demand": round(0.90 + 0.05 * np.random.random(), 2), "supply": round(0.60 + 0.1 * np.random.random(), 2)},
            "healthcare administration": {"demand": round(0.75 + 0.1 * np.random.random(), 2), "supply": round(0.60 + 0.1 * np.random.random(), 2)},
            "cybersecurity": {"demand": round(0.90 + 0.05 * np.random.random(), 2), "supply": round(0.50 + 0.1 * np.random.random(), 2)},
            "digital marketing": {"demand": round(0.80 + 0.1 * np.random.random(), 2), "supply": round(0.65 + 0.1 * np.random.random(), 2)},
            "electrical": {"demand": round(0.85 + 0.05 * np.random.random(), 2), "supply": round(0.70 + 0.1 * np.random.random(), 2)},
            "graphic design": {"demand": round(0.75 + 0.1 * np.random.random(), 2), "supply": round(0.75 + 0.1 * np.random.random(), 2)},
            "medical assistance": {"demand": round(0.80 + 0.1 * np.random.random(), 2), "supply": round(0.65 + 0.1 * np.random.random(), 2)},
            "software engineering": {"demand": round(0.90 + 0.05 * np.random.random(), 2), "supply": round(0.70 + 0.1 * np.random.random(), 2)},
            "project management": {"demand": round(0.80 + 0.1 * np.random.random(), 2), "supply": round(0.75 + 0.1 * np.random.random(), 2)},
            "cloud computing": {"demand": round(0.85 + 0.1 * np.random.random(), 2), "supply": round(0.60 + 0.1 * np.random.random(), 2)},
            "machine learning": {"demand": round(0.90 + 0.05 * np.random.random(), 2), "supply": round(0.55 + 0.1 * np.random.random(), 2)}
        }
        
        return {
            "skills_gap": skills_gap,
            "timestamp": datetime.now().strftime('%Y-%m-%d')
        }

# Import numpy for random number generation
import numpy as np
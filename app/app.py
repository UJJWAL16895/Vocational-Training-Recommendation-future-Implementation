from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the recommendation engine
try:
    from models.recommendation_engine import RecommendationEngine
    recommendation_engine = RecommendationEngine()
    model_loaded = True
except (ImportError, FileNotFoundError):
    # If the model isn't available yet, we'll still run the app
    # but won't provide real recommendations
    model_loaded = False

app = Flask(__name__)

# Configure app from environment variables
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')
app.config['DEBUG'] = os.environ.get('DEBUG', 'True').lower() == 'true'

# Sample data for demonstration purposes
# In a real implementation, this would come from your trained models
sample_training_programs = [
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
        }
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
        }
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
        }
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
        }
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
        }
    },
    {
        "id": 6,
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
        }
    },
    {
        "id": 7,
        "name": "Nursing Assistant Certificate",
        "duration": "12 weeks",
        "skills": ["Patient Care", "Medical Terminology", "First Aid", "CPR"],
        "job_placement_rate": 90,
        "cost": "$6,000",
        "demand_score": 0.88,
        "features": {
            "technology": 0.2,
            "healthcare": 0.9,
            "trades": 0.1,
            "business": 0.2,
            "creative": 0.1,
            "education": 0.3
        }
    },
    {
        "id": 8,
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
        }
    }
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # Get user input from the form
        user_data = {
            'interests': request.form.getlist('interests'),
            'prior_experience': request.form.get('prior_experience'),
            'education_level': request.form.get('education_level'),
            'preferred_duration': request.form.get('preferred_duration'),
            'region': request.form.get('region')
        }
        
        # Generate recommendations
        if model_loaded:
            # Use the actual recommendation engine when available
            recommendations = recommendation_engine.get_recommendations(user_data)
        else:
            # Use sample data for demonstration
            # In a real implementation, you would use your ML model here
            recommendations = get_sample_recommendations(user_data)
        
        return render_template('recommendations.html', 
                               recommendations=recommendations,
                               user_data=user_data)
    
    # If GET request, show the recommendation form
    return render_template('recommend_form.html')

@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    """API endpoint for getting recommendations"""
    user_data = request.json
    
    if model_loaded:
        recommendations = recommendation_engine.get_recommendations(user_data)
    else:
        recommendations = get_sample_recommendations(user_data)
    
    return jsonify({
        'success': True,
        'recommendations': recommendations
    })

@app.route('/market-trends')
def market_trends():
    """Display market trends and employment data visualizations"""
    # In a real implementation, you would load and process actual data here
    return render_template('market_trends.html')

@app.route('/weather')
def weather():
    """Display weather information for farming"""
    return render_template('weather.html')

@app.route('/api/market-trends')
def api_market_trends():
    """API endpoint for market trends data"""
    # Get the region from query parameters, default to 'all'
    region = request.args.get('region', 'all')
    
    # Import the API client for real-time data
    from data.api_client import EmploymentDataClient
    
    try:
        # Create API client
        data_client = EmploymentDataClient()
        
        # Fetch real-time market trends data
        market_trends = data_client.fetch_market_trends(region)
        
        # If we have real-time data, use it
        if market_trends and 'regional_comparison' in market_trends:
            employment_data = market_trends['regional_comparison']
        elif model_loaded:
            # Fall back to recommendation engine data if API fails
            employment_data = recommendation_engine.employment_data
        else:
            # Fall back to sample data if both API and model fail
            employment_data = {
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
        
        # Fetch real-time skills gap data
        skills_data = data_client.fetch_skills_gap_data()
        
        # If we have real-time skills gap data, use it
        if skills_data and 'skills_gap' in skills_data:
            skills_gap = skills_data['skills_gap']
        else:
            # Fall back to sample skills gap data
            skills_gap = {
                "web development": {"demand": 0.88, "supply": 0.72},
                "data analysis": {"demand": 0.92, "supply": 0.65},
                "healthcare administration": {"demand": 0.75, "supply": 0.60},
                "cybersecurity": {"demand": 0.95, "supply": 0.55},
                "digital marketing": {"demand": 0.82, "supply": 0.70},
                "electrical": {"demand": 0.88, "supply": 0.75},
                "graphic design": {"demand": 0.78, "supply": 0.80},
                "medical assistance": {"demand": 0.85, "supply": 0.68}
            }
        
        # Get job growth projection data from market trends if available
        if market_trends and 'job_growth' in market_trends:
            job_growth = market_trends['job_growth']
        else:
            # Fall back to sample job growth data
            job_growth = {
                "years": ["2023", "2024", "2025", "2026", "2027"],
                "industries": {
                    "technology": [5.2, 6.8, 8.5, 10.2, 12.0],
                    "healthcare": [4.5, 5.7, 7.0, 8.2, 9.5],
                    "trades": [3.0, 3.8, 4.5, 5.2, 6.0],
                    "business": [3.8, 4.5, 5.2, 6.0, 6.8],
                    "creative": [2.5, 3.2, 4.0, 4.8, 5.5],
                    "education": [2.0, 2.5, 3.0, 3.5, 4.0]
                }
            }
    except Exception as e:
        print(f"Error fetching real-time data: {e}")
        # Fall back to recommendation engine data if API fails
        if model_loaded:
            employment_data = recommendation_engine.employment_data
        else:
            # Sample employment data if model is not loaded
            employment_data = {
                "northeast": {
                    "technology": 0.85,
                    "healthcare": 0.78,
                    "trades": 0.65,
                    "business": 0.82,
                    "creative": 0.75,
                    "education": 0.70
                },
                # ... other regions ...
            }
        
        # Sample skills gap data
        skills_gap = {
            "web development": {"demand": 0.88, "supply": 0.72},
            "data analysis": {"demand": 0.92, "supply": 0.65},
            "healthcare administration": {"demand": 0.75, "supply": 0.60},
            "cybersecurity": {"demand": 0.95, "supply": 0.55},
            "digital marketing": {"demand": 0.82, "supply": 0.70},
            "electrical": {"demand": 0.88, "supply": 0.75},
            "graphic design": {"demand": 0.78, "supply": 0.80},
            "medical assistance": {"demand": 0.85, "supply": 0.68}
        }
        
        # Sample job growth projection data
        job_growth = {
            "years": ["2023", "2024", "2025", "2026", "2027"],
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
    response_data = {}
    
    # If a specific region is requested
    if region != 'all' and region in employment_data:
        response_data['industry_demand'] = employment_data[region]
    else:
        # For 'all', calculate average across regions
        avg_demand = {}
        for industry in employment_data['northeast'].keys():
            total = sum(region_data[industry] for region_data in employment_data.values())
            avg_demand[industry] = total / len(employment_data)
        response_data['industry_demand'] = avg_demand
    
    # Include regional comparison data
    response_data['regional_comparison'] = employment_data
    
    # Include skills gap data
    response_data['skills_gap'] = skills_gap
    
    # Include job growth data
    response_data['job_growth'] = job_growth
    
    return jsonify(response_data)

def get_sample_recommendations(user_data):
    """Generate sample recommendations based on user data
    This is a placeholder until the actual ML model is implemented
    """
    # Simple filtering based on user interests
    recommendations = []
    
    # Convert interests to lowercase for case-insensitive matching
    user_interests = [interest.lower() for interest in user_data.get('interests', [])]
    
    # First, identify programs that directly match user interests based on features
    matching_programs = []
    other_programs = []
    
    for program in sample_training_programs:
        program_copy = program.copy()
        
        # Calculate a more sophisticated match score
        match_score = 0
        direct_category_match = False
        
        # Check for direct category matches in features
        if 'features' in program:
            for interest in user_interests:
                if interest in program['features'] and program['features'][interest] > 0.7:
                    match_score += 3  # Strong boost for direct category match
                    direct_category_match = True
                elif interest in program['features'] and program['features'][interest] > 0.4:
                    match_score += 1  # Smaller boost for partial category match
        
        # Also check skills for additional matching
        program_skills = [skill.lower() for skill in program['skills']]
        for interest in user_interests:
            for skill in program_skills:
                if interest in skill or skill in interest:
                    match_score += 1
        
        # Add match score to the program
        program_copy['match_score'] = match_score
        
        # Separate into direct matches and other programs
        if direct_category_match:
            matching_programs.append(program_copy)
        else:
            other_programs.append(program_copy)
    
    # Sort each group by match score and demand score
    matching_programs.sort(key=lambda x: (x['match_score'], x['demand_score']), reverse=True)
    other_programs.sort(key=lambda x: (x['match_score'], x['demand_score']), reverse=True)
    
    # Combine the groups, with matching programs first
    recommendations = matching_programs + other_programs
    
    # If no specific interests were provided, include all programs
    if not user_interests:
        recommendations = [program.copy() for program in sample_training_programs]
        for program in recommendations:
            program['match_score'] = 1
        recommendations.sort(key=lambda x: x['demand_score'], reverse=True)
    
    return recommendations

if __name__ == '__main__':
    # Get port from environment variable (useful for Heroku/other cloud platforms)
    port = int(os.environ.get('PORT', 5000))
    # Run the app, binding to all interfaces (0.0.0.0) for cloud deployment
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
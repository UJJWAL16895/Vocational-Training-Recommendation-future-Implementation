from app.app import app

# This file serves as the entry point for Vercel serverless functions
# It simply imports the Flask app instance from the main application

# Vercel will automatically detect this file and use it as the entry point
# The app object is imported from app.app as defined in the vercel.json configuration
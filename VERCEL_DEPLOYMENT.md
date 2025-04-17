# Deploying LODA to Vercel

This guide provides step-by-step instructions for deploying the LODA application to Vercel.

## Prerequisites

1. A [Vercel](https://vercel.com) account
2. [Vercel CLI](https://vercel.com/docs/cli) installed (optional, for local development)

## Deployment Steps

### 1. Prepare Your Project

The project has already been configured for Vercel deployment with the following files:

- `vercel.json` - Configuration file for Vercel deployment
- `api/index.py` - Serverless function entry point
- `requirements-vercel.txt` - Dependencies for Vercel deployment

### 2. Deploy to Vercel

#### Option 1: Using Vercel Dashboard

1. Log in to your [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your Git repository or upload your project files
4. Configure the project:
   - Framework Preset: Other
   - Build Command: Leave empty
   - Output Directory: Leave empty
   - Install Command: `pip install -r requirements-vercel.txt`
5. Add environment variables from your `.env.production` file
6. Click "Deploy"

#### Option 2: Using Vercel CLI

1. Install Vercel CLI: `npm i -g vercel`
2. Navigate to your project directory
3. Run `vercel login` and follow the prompts
4. Run `vercel` to deploy
5. Follow the prompts to configure your project

### 3. Environment Variables

Make sure to set these environment variables in your Vercel project settings:

- `FLASK_ENV`: Set to `production`
- `FLASK_DEBUG`: Set to `False`
- `DEBUG`: Set to `False`
- `SECRET_KEY`: Generate a secure random key

### 4. Verify Deployment

Once deployed, Vercel will provide you with a URL to access your application. Visit this URL to verify that your application is working correctly.

### 5. Custom Domain (Optional)

To use a custom domain with your Vercel deployment:

1. Go to your project in the Vercel Dashboard
2. Navigate to "Settings" > "Domains"
3. Add your custom domain and follow the instructions

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required packages are listed in `requirements-vercel.txt`
2. **Environment Variables**: Check that all required environment variables are set in Vercel
3. **Static Files**: Verify that static files are being served correctly

If you encounter issues, check the Vercel deployment logs for error messages.
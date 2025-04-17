# Deployment Guide for LODA (Learning Opportunity and Development Assistant)

This guide provides step-by-step instructions for deploying the LODA application to various platforms.

## Table of Contents
1. [Preparation Steps](#preparation-steps)
2. [Heroku Deployment](#heroku-deployment)
3. [Docker Deployment](#docker-deployment)
4. [AWS Elastic Beanstalk Deployment](#aws-elastic-beanstalk-deployment)
5. [Azure Web App Deployment](#azure-web-app-deployment)
6. [Troubleshooting](#troubleshooting)

## Preparation Steps

Before deploying to any platform, complete these preparation steps:

1. **Update Environment Variables**
   - Edit the `.env` file to set production values:
     ```
     FLASK_ENV=production
     FLASK_DEBUG=False
     DEBUG=False
     SECRET_KEY=your-secure-random-key
     ```
   - Generate a secure random key for `SECRET_KEY` using Python:
     ```python
     import secrets
     print(secrets.token_hex(16))
     ```

2. **Verify Requirements**
   - Ensure all dependencies are listed in `requirements.txt`
   - The current file already includes necessary dependencies

3. **Test Locally in Production Mode**
   - Set environment variables for production
   - Run the application: `python -m app.app`
   - Verify everything works correctly

## Heroku Deployment

Heroku is one of the simplest platforms for deploying Flask applications.

1. **Install Heroku CLI**
   - Download and install from [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

2. **Login to Heroku**
   ```
   heroku login
   ```

3. **Create a Heroku App**
   ```
   heroku create loda-app
   ```
   Or use a custom name of your choice

4. **Set Environment Variables**
   ```
   heroku config:set FLASK_ENV=production
   heroku config:set FLASK_DEBUG=False
   heroku config:set DEBUG=False
   heroku config:set SECRET_KEY=your-secure-random-key
   ```

5. **Deploy to Heroku**
   ```
   git add .
   git commit -m "Prepare for deployment"
   git push heroku main
   ```
   If your branch is not `main`, use: `git push heroku your-branch-name:main`

6. **Scale the Dyno**
   ```
   heroku ps:scale web=1
   ```

7. **Open the App**
   ```
   heroku open
   ```

## Docker Deployment

1. **Create a Dockerfile**
   Create a file named `Dockerfile` in the root directory with the following content:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   ENV FLASK_ENV=production
   ENV FLASK_DEBUG=False
   ENV DEBUG=False
   ENV SECRET_KEY=your-secure-random-key

   EXPOSE 8000

   CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app.app:app"]
   ```

2. **Build the Docker Image**
   ```
   docker build -t loda-app .
   ```

3. **Run the Docker Container**
   ```
   docker run -p 8000:8000 loda-app
   ```

4. **Access the Application**
   Open your browser and navigate to `http://localhost:8000`

## AWS Elastic Beanstalk Deployment

1. **Install AWS CLI and EB CLI**
   - [AWS CLI Installation](https://aws.amazon.com/cli/)
   - [EB CLI Installation](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html)

2. **Configure AWS Credentials**
   ```
   aws configure
   ```

3. **Initialize EB Application**
   ```
   eb init -p python-3.9 loda-app
   ```

4. **Create an Environment**
   ```
   eb create loda-production
   ```

5. **Set Environment Variables**
   ```
   eb setenv FLASK_ENV=production FLASK_DEBUG=False DEBUG=False SECRET_KEY=your-secure-random-key
   ```

6. **Deploy the Application**
   ```
   eb deploy
   ```

7. **Open the Application**
   ```
   eb open
   ```

## Azure Web App Deployment

1. **Install Azure CLI**
   - [Azure CLI Installation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

2. **Login to Azure**
   ```
   az login
   ```

3. **Create a Resource Group**
   ```
   az group create --name loda-resource-group --location eastus
   ```

4. **Create an App Service Plan**
   ```
   az appservice plan create --name loda-service-plan --resource-group loda-resource-group --sku B1
   ```

5. **Create a Web App**
   ```
   az webapp create --name loda-webapp --resource-group loda-resource-group --plan loda-service-plan --runtime "PYTHON|3.9"
   ```

6. **Set Environment Variables**
   ```
   az webapp config appsettings set --name loda-webapp --resource-group loda-resource-group --settings FLASK_ENV=production FLASK_DEBUG=False DEBUG=False SECRET_KEY=your-secure-random-key
   ```

7. **Deploy the Application**
   ```
   az webapp up --name loda-webapp --resource-group loda-resource-group
   ```

8. **Access the Application**
   ```
   az webapp browse --name loda-webapp --resource-group loda-resource-group
   ```

## Troubleshooting

### Common Issues

1. **Application Error on Heroku**
   - Check logs: `heroku logs --tail`
   - Ensure Procfile is correct: `web: gunicorn app.app:app`
   - Verify all dependencies are in requirements.txt

2. **Missing Environment Variables**
   - Verify all required environment variables are set
   - Check for typos in variable names

3. **Database Connection Issues**
   - If you add a database later, ensure connection strings are properly configured
   - Use environment variables for database credentials

4. **Static Files Not Loading**
   - Ensure static files are properly configured in Flask
   - Check file paths and permissions

5. **Server Timeout**
   - Optimize slow operations
   - Consider using background workers for long-running tasks

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs of your deployment platform
2. Consult the platform-specific documentation
3. Search for similar issues on Stack Overflow
4. Reach out to the platform's support channels
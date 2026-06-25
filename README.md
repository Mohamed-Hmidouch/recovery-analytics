# Smart Recovery ML

This project provides an API and machine learning pipeline for recovery analytics.

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (if you wish to run the project locally without Docker)

## Setup and Running the Project

The easiest way to run the project is using Docker Compose, which sets up both the FastAPI application and the PostgreSQL database.

1. **Environment Variables**: Ensure you have a `.env` file in the root directory. You can use the following variables as an example:
   ```env
   POSTGRES_USER=myuser
   POSTGRES_PASSWORD=mypassword
   POSTGRES_DB=mydb
   API_PORT=8000
   ```

2. **Run with Docker**:
   Start the services in detached mode:
   ```bash
   docker-compose up -d --build
   ```

3. **Access the API**:
   Once the containers are running, the API documentation (Swagger UI) will be available at:
   [http://localhost:8000/docs](http://localhost:8000/docs)

4. **Stop the services**:
   ```bash
   docker-compose down
   ```

## Running Locally (Without Docker)

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies (Java 17 is also required for PySpark):
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure your local PostgreSQL database is running and the `.env` variables match your local DB configuration.

4. Run the API:
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

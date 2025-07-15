FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /app

# Install requirements separately for better cache usage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy your code
COPY . .

CMD ["python", "--version"]


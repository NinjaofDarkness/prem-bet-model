FROM python:3.13.5-slim

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /app
COPY . .
CMD ["python", "--version"]


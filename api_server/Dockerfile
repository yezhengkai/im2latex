# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
FROM python:3.8-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY api_server/requirements.txt ./requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

# Copy only the relevant directories
COPY im2latex/ ./im2latex
COPY api_server/ ./api
COPY data/processed/im2latex_100k/*.json data/processed/im2latex_100k/

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
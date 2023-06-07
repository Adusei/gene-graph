FROM jupyter/scipy-notebook:latest

# Install additional packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your notebooks or project files
COPY GeneGraph.ipynb /home/jovyan/work/

# Set the working directory
WORKDIR /home/jovyan/work/

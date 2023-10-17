FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the Flask application code into the container
COPY . .

# Expose the port your Flask app will run on
EXPOSE 8080

# Define the command to run your Flask app
CMD ["python3", "app.py"]
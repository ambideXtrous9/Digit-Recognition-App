FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the /app directory in the container
COPY . .


RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port you want your app on
EXPOSE 8601

CMD ["streamlit", "run", "app.py", "–server.port=8601", "–server.address=0.0.0.0"]

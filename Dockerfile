# # Use an official base image
FROM python:3.9-slim

# # Set the working directory in the container
WORKDIR /code

# # Copy the current directory contents into the container at /app
COPY . /code
# ENV DEBIAN_FRONTEND=noninteractive
# ENV OLLAMA_MODEL=phi3:mini



# # Install any needed packages specified in requirements.txt
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gnupg2 \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# RUN curl https://ollama.ai/install.sh | sh

RUN pip install --no-cache-dir -r /code/requirements.txt


#     # Expose the default Ollama port
# EXPOSE 80  11434

#     # Create a startup script
# RUN echo '#!/bin/sh' > /start.sh && \
#         echo 'ollama serve &' >> /start.sh && \
#         echo 'sleep 10' >> /start.sh && \
#         echo 'ollama pull $OLLAMA_MODEL || echo "Failed to pull model $OLLAMA_MODEL"' >> /start.sh && \
#         echo 'uvicorn api:app --host 0.0.0.0 --port 80' >> /start.sh && \
#         chmod +x /start.sh
    
#     # Command to run the application
# CMD ["/start.sh"]
    
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","80"]
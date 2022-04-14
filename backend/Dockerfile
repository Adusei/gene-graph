FROM python:3.9.6

# Copy local code to the container image.
ADD requirements.txt /

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies.
RUN pip install -U pip==20.1.1

RUN pip install -r /requirements.txt



# Copy local code to the container image.
COPY . ./app


# Run the web service on container startup.
EXPOSE 8000
CMD exec python server.py

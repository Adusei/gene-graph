FROM python:3.9.6

# Copy local code to the container image.
ENV APP_HOME /srv/app
WORKDIR $APP_HOME

# Install production dependencies.
RUN pip install -U pip==20.1.1

# Copy local code to the container image.
COPY . .

RUN pip install -r requirements.txt

# Run the web service on container startup.
EXPOSE 8000
CMD exec python -m src.server

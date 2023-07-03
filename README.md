# Gene Graph

An analysis into how gene networks help us understand Neurodegenerative diseases + is abstracted to analyze any disease or disease hierarchy in the [CTD](https://ctdbase.org/)

For more Information see [the post on medium](https://medium.com/@dingeej/using-machine-learning-to-understand-the-gene-network-associations-of-neurodegenerative-disease-a96d6997fb6)

## Spin Up The Notebook
```
docker-compose up
```

## Run The Tests
```
docker exec -it $(docker ps --filter ancestor=gene-graph_jupyter -q) python -m unittest discover
```

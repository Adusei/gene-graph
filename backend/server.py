
import os
import pandas as pd
from flask import Flask

from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app) ## REMOVE once dockerized

@app.route("/")
def hello_world():
    return "<p>Hello, Worl!!d!</p>"

@app.route("/sample_data")
def sample_data():
    base  = {"nodes":[{"id":0},{"id":1},{"id":2},{"id":3},{"id":4},{"id":5},{"id":6},{"id":7},{"id":8},{"id":9},{"id":10},{"id":11},{"id":12},{"id":13},{"id":14},{"id":15},{"id":16},{"id":17},{"id":18},{"id":19},{"id":20},{"id":21},{"id":22},{"id":23},{"id":24},{"id":25},{"id":26},{"id":27},{"id":28},{"id":29},{"id":30},{"id":31},{"id":32},{"id":33},{"id":34},{"id":35},{"id":36},{"id":37},{"id":38},{"id":39},{"id":40},{"id":41},{"id":42},{"id":43},{"id":44},{"id":45},{"id":46},{"id":47},{"id":48},{"id":49}],"links":[{"source":1,"target":0},{"source":2,"target":0},{"source":3,"target":1},{"source":4,"target":0},{"source":5,"target":0},{"source":6,"target":3},{"source":7,"target":4},{"source":8,"target":2},{"source":9,"target":5},{"source":10,"target":8},{"source":11,"target":1},{"source":12,"target":2},{"source":13,"target":7},{"source":14,"target":10},{"source":15,"target":7},{"source":16,"target":8},{"source":17,"target":4},{"source":18,"target":0},{"source":19,"target":9},{"source":20,"target":8},{"source":21,"target":18},{"source":22,"target":21},{"source":23,"target":19},{"source":24,"target":22},{"source":25,"target":14},{"source":26,"target":22},{"source":27,"target":18},{"source":28,"target":26},{"source":29,"target":2},{"source":30,"target":5},{"source":31,"target":26},{"source":32,"target":24},{"source":33,"target":21},{"source":34,"target":26},{"source":35,"target":5},{"source":36,"target":6},{"source":37,"target":15},{"source":38,"target":32},{"source":39,"target":31},{"source":40,"target":11},{"source":41,"target":25},{"source":42,"target":19},{"source":43,"target":15},{"source":44,"target":34},{"source":45,"target":41},{"source":46,"target":37},{"source":47,"target":2},{"source":48,"target":3},{"source":49,"target":35}]}

    df = pd.read_csv('parkinsons_network.csv', delimiter='|')
    nodes = []
    links = []

    unique_nodes = list(df['FROM'].unique()) + list(df['TO'].unique())
    for n in unique_nodes:
        nodes.append({"id": n})

    for ix, row in df.iterrows():
        links.append({"source": row.FROM, "target": row.TO})

    return {"nodes": nodes, "links": links}


if __name__ == "__main__":
    # app = init_app()
    app.run(
        host="0.0.0.0",
        debug = True,
        port=int(os.getenv("PORT", 8000)),
        # auto_reload=os.getenv("SANIC_AUTORELOAD", False),
    )

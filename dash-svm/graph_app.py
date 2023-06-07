import json

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import dash_cytoscape as cyto
# import utils.dash_reusable_components as drc
import pandas as pd

app = dash.Dash(__name__)
server = app.server

# DRUG_IMG = 'http://ctdbase.org/cas-images/4/1199-18-4.v16437.png'

# ###################### DATA PREPROCESSING ######################
# Load data

# note get the MESH file from et_df('CTD_chemicals_diseases') and filter to
# whatever disease you need

def get_color_scale(df_len):
    from colour import Color
    start = "#1aff00"
    end = "#ffffff"
    colors = list(Color(start).range_to(Color(end),df_len))
    return colors

def get_graph_elements(disease_id):

    # df = pd.read_csv('../gene_importance_df.csv')[:50]
    df = pd.read_csv('data/MESH:{disease_id}.csv'.format(disease_id=disease_id))
    df = df.sort_values('InferenceScore', ascending=False)[:100]
    colors = get_color_scale(len(df))
    df['line_color'] = colors

    nodes = set()

    cy_edges = []
    cy_nodes = []

    for ix, (network_edge) in df.iterrows():
        source = network_edge.ChemicalName
        target = network_edge.InferenceGeneSymbol
        score = network_edge.InferenceScore
        pub_med_id = network_edge.PubMedIDs

        line_color = network_edge.line_color.hex # '#fc0505'

        if source not in nodes:
            nodes.add(source)
            cy_nodes.append({"data": {"id": source, "label": source}})
        if target not in nodes:
            nodes.add(target)
            cy_nodes.append({"data": {"id": target, "label": target}})

        cy_edges.append({
            'data': {
                'id': source + '-' + target,
                'source': source,
                'target': target,
                'color': line_color,
                'pub_med_id': pub_med_id
            }
        })

    return cy_edges + cy_nodes

default_stylesheet = [
    {
        "selector": 'node',
        'style': {
            "color": "#FF4136",
            "opacity": 0.65,
            "label": "data(id)"
        }
    },
    {
        "selector": 'edge',
        'style': {
            "curve-style": "bezier",
            "opacity": 0.65,
            "line-color": "data(color)"
        }
    },
]

styles = {
    'json-output': {
        'overflow-y': 'scroll',
        'height': 'calc(50% - 25px)',
        'border': 'thin lightgrey solid'
    },
    'tab': {
        'height': 'calc(98vh - 105px)'
    }
}

def get_cyto_elements():
    elements=cy_edges + cy_nodes
    return elements

def get_cyto():
    return cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'circle'},
        elements=[],#get_graph_elements('D009369'),
        style={
            'height': '95vh',
            'width': '100%'
        }
    )


app.layout = html.Div([
    html.Div(className='eight columns', children=[
        get_cyto()
    ]),

    html.Div(className='four columns', children=[
        # dcc.Tabs(id='tabs', children=[
            # dcc.Tab(label='Control Panel', children=[
                html.P(
                    'Select Disease',
                    id='disease-select-para'
                ),
                dcc.Dropdown(
                   # name='Select Disease',
                   id='dropdown-disease-select',
                   options=[
                       {'label': 'Cancer', 'value': 'D009369'},
                       {'label': 'Parkinsons Disease', 'value': 'D010300'}
                   ],
                   value='D010300'
                ),
                dcc.Dropdown(
                    id='dropdown-layout',
                    value='circle',
                    clearable=False
                ),
                html.A('',id='pubchem-link', href='', target='_blank'),
                html.P(
                    'this is a para',
                    id='abstract-para'
                )
        # ]),
    ])
])


@app.callback(Output('cytoscape', 'layout'),
              [Input('dropdown-layout', 'value')])
def update_cytoscape_layout(layout):
    return {'name': layout}

@app.callback(Output('cytoscape', 'elements'),
              [Input('dropdown-disease-select', 'value')])
def update_disease_layout(disease_id):
    elements = get_graph_elements(disease_id)
    return elements


@app.callback([
               Output('cytoscape', 'stylesheet'),
               Output('abstract-para', 'children'),
               Output('pubchem-link', 'children'),
               Output('pubchem-link', 'href')
              ],
              [Input('cytoscape', 'tapNode')])
def generate_stylesheet(node):
    node_shape = 'circle'
    follower_color = '#03fc28'
    following_color = '#B10DC9'

    if not node:
        return default_stylesheet, '' ,'' ,''

    stylesheet = [{
        "selector": 'node',
        'style': {
            'opacity': 0.3,
            'shape': node_shape
        }
    }, {
        'selector': 'edge',
        'style': {
            'opacity': 0.2,
            "curve-style": "bezier",
        }
    }, {
        "selector": 'node[id = "{}"]'.format(node['data']['id']),
        "style": {
            'background-color': following_color,
            "border-color": "purple",
            "border-width": 2,
            "border-opacity": 1,
            "opacity": 1,

            "label": "data(label)",
            "color": following_color,
            "text-opacity": 1,
            "font-size": 12,
            'z-index': 9999
        }
    }]

    first_pub_med_link = None
    for edge in node['edgesData']:
        print('edge: ', edge.get('pub_med_id'))
        if not first_pub_med_link:
            first_pub_med_link = edge.get('pub_med_id').split('|')[0]

        if edge['source'] == node['data']['id']:
            stylesheet.append({
                "selector": 'node[id = "{}"]'.format(edge['target']),
                "style": {
                    'background-color': following_color,
                    'opacity': 0.9,
                    'label': "data(label)",
                }
            })
            stylesheet.append({
                "selector": 'edge[id= "{}"]'.format(edge['id']),
                "style": {
                    "mid-target-arrow-color": following_color,
                    "mid-target-arrow-shape": "vee",
                    "line-color": following_color,
                    'opacity': 0.9,
                    'z-index': 5000
                }
            })

        if edge['target'] == node['data']['id']:
            stylesheet.append({
                "selector": 'node[id = "{}"]'.format(edge['source']),
                "style": {
                    'background-color': follower_color,
                    'opacity': 0.9,
                    'z-index': 9999,
                    "color": follower_color,
                    'label': "data(label)"
                }
            })
            stylesheet.append({
                "selector": 'edge[id= "{}"]'.format(edge['id']),
                "style": {
                    "mid-target-arrow-color": follower_color,
                    "mid-target-arrow-shape": "vee",
                    "line-color": follower_color,
                    "text-color": follower_color,
                    'opacity': 1,
                    'z-index': 5000
                }
            })

    from parse_nih_html import main as get_nih_html
    paper_title, abstract_paragraph, paper_url = get_nih_html(first_pub_med_link)

    return stylesheet, abstract_paragraph[:300] + '...', paper_title, paper_url



if __name__ == '__main__':
    app.run_server(debug=True)

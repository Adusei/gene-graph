# Gene Graph

An analysis into how gene networks help us understand Neurodegenerative diseases

For more Information see the post on medium here https://medium.com/p/a96d6997fb6/edit

## TO DO
- Set up a password in the docker compose thing
- name the docker image
- human doc strings
- gitignore
- new repo


## Spin Up The Notebook
```
docker-compose up
```


## Run The Tests
```

```



old notebook
https://github.com/Adusei/gene-graph/blob/5a8ba52027d118fefea5a4c70868a3e116fff97a/notebook/GeneGraph.ipynb

masters doc
https://docs.google.com/document/d/1vWF39uO3U2cma6IjJ3qls5WtBD0g-GdbDshnNFpSGSI/edit


```
from src.disease_classifier import DiseaseClassifier
from ctd import get_data
input_df = get_data('ChemicalDiseaseInteractions')
kw = {'input_df': input_df, 'parent_disease': 'MESH:D019636', 'gene_count': 10, 'show_plots': False, 'use_class_weights': True, 'oversample': False,'show_feature_importance': True, 'model_type': 'LINEAR'}
dc = DiseaseClassifier(**kw)
model_metrics, model, X_train = dc.main()
```
<!-- https://github.com/slundberg/shap/issues/1110#issuecomment-720337500
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
explainer = shap.KernelExplainer(data=np.array(X_train), feature_names=X_train.columns, model=model)

shap_values = explainer.shap_values(X_train.iloc[50,:], nsamples=100)

X = X_train.iloc[50,:]

masker = shap.maskers.Independent(X_train, 10)
explainer = shap.KernelExplainer(model.predict, masker.data)

shap_values = explainer.shap_values(X)
idx = 0
shap.force_plot(explainer.expected_value[0], shap_values[0][idx], masker.data[idx]) -->

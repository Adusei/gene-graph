import os
import gzip
import shutil
import requests

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import Sequential, metrics
from keras.layers import Input, Dense, BatchNormalization, LSTM, Embedding, Bidirectional, Normalization, Conv1D, Dropout, MaxPool2D, MaxPooling1D, Flatten
from keras.models import Model

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ctd.utils import get_disease_hierarchy


VERBOSE=False
EPOCHS=25
TRAIN_SIZE=.75

METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'),
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc'),
      metrics.AUC(name='prc', curve='PR')
]

from typing import List

class DiseaseClassifier:
    def __init__(
        self,
        input_df: pd.DataFrame,
        parent_disease: str,
        gene_count: int,
        show_plots: bool,
        use_class_weights: bool,
        oversample: bool,
        classification: str = 'binary'
    ) -> None:
        """
        Initialize the DiseaseClassifier class.

        Args:
            input_df: The input dataframe containing the data for classification.
            parent_disease: The parent disease for which classification is performed.
            gene_count: The number of genes to consider for classification.
            show_plots: Flag indicating whether to show plots or not.
            use_class_weights: Flag indicating whether to use class weights or not.
            oversample: Flag indicating whether to perform oversampling or not.
            classification: The type of classification, default is 'binary'.

        Raises:
            Exception: If both use_class_weights and oversample are True.

        Returns:
            None
        """
        self.input_df = input_df
        self.parent_disease = parent_disease
        self.target_diseases: List[str] = get_disease_hierarchy(self.parent_disease)
        self.gene_count = gene_count
        self.show_plots = show_plots
        self.stop_early = True
        self.use_class_weights = use_class_weights
        self.oversample = oversample
        self.classification = classification
        self.top_n_genes = self.get_genes()
        self.model_type = 'DNN'

        if self.use_class_weights and self.oversample:
            raise Exception('Need to either use class weights OR oversample')

    def get_genes(self) -> List[str]:
        """
        Get the top N most frequent genes from the input DataFrame.

        Returns:
            List[str]: List of top N genes.
        """
        gene_df = pd.DataFrame(self.input_df.groupby(['InferenceGeneSymbol']).size()).reset_index()
        gene_df.columns = ['InferenceGeneSymbol', 'cnt']
        top_n_genes_df = gene_df.sort_values('cnt', ascending=False)[:self.gene_count]
        top_n_genes = top_n_genes_df['InferenceGeneSymbol'].unique()

        return top_n_genes

    def prep_training_data(self, one_hot: bool) -> pd.DataFrame:
        """
        Prepare the training data for the DiseaseClassifier by taking the
        input dataframe and pivoting it such that we onehot encode the genes

        Args:
            one_hot: Use the one hot encodings otherwise it will fill the values
            of the gene features with the inference score

        Returns:
            DataFrame: The prepared training data.
        """

        gene_df = self.input_df.loc[self.input_df['DirectEvidence'].isnull()][['ChemicalName', 'DiseaseName', 'InferenceGeneSymbol', 'InferenceScore', 'DiseaseID']]
        gene_df = gene_df.loc[gene_df['InferenceGeneSymbol'].isin(self.top_n_genes)]

        evidence_df = self.input_df.loc[self.input_df['DirectEvidence'].notnull()][['ChemicalName', 'DiseaseName', 'DirectEvidence', 'DiseaseID']]

        if one_hot:
            dummy_df = pd.get_dummies(gene_df, prefix='', prefix_sep='',columns=['InferenceGeneSymbol'])

            gb_df = dummy_df.groupby(['DiseaseName', 'ChemicalName', 'DiseaseID']).agg({np.max}).reset_index()
            gb_df.columns = gb_df.columns.droplevel(1)

        else:
            gb_df = gene_df.pivot_table(index=['DiseaseName', 'ChemicalName', 'DiseaseID'],
                              columns='InferenceGeneSymbol',
                              values='InferenceScore',
                              aggfunc='max',
                              fill_value=0).reset_index()

        merged_df = gb_df.merge(evidence_df, on=['ChemicalName', 'DiseaseName', 'DiseaseID'])

        # merged_df['label'] = np.where(merged_df['DirectEvidence'] == 'marker/mechanism',
        #                  merged_df['InferenceScore'] * -1,
        #                  merged_df['InferenceScore'])


        return merged_df

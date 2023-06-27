import os
import gzip
import shutil
import requests
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ## TO DO set in docker
import tensorflow as tf
from keras import Sequential, metrics
from keras.layers import Input, Dense, BatchNormalization, LSTM, Embedding, Bidirectional, Normalization, Conv1D, Dropout, MaxPool2D, MaxPooling1D, Flatten
from keras.models import Model
from imblearn.over_sampling import SMOTE

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

class DiseaseClassifier:
    def __init__(
        self,
        input_df: pd.DataFrame,
        parent_disease: str,
        gene_count: int,
        show_plots: bool,
        use_class_weights: bool,
        oversample: bool,
        classification: str = 'binary',
        use_gene_inference_score: bool = False
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
            use_gene_inference_score: if True we use the score for the features,
            otherwise we use onehot encoding to represent a link exists
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
        self.use_gene_inference_score = use_gene_inference_score
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

    def prep_training_data(self) -> pd.DataFrame:
        """
        Prepare the training data for the DiseaseClassifier by taking the
        input dataframe and pivoting it such that we onehot encode the genes
        or add the inference score of the gene as the feature value for the
        observation

        Args:
            one_hot: Use the one hot encodings otherwise it will fill the values
            of the gene features with the inference score

        Returns:
            DataFrame: The prepared training data.
        """

        gene_df = self.input_df.loc[self.input_df['DirectEvidence'].isnull()][['ChemicalName', 'DiseaseName', 'InferenceGeneSymbol', 'InferenceScore', 'DiseaseID']]
        gene_df = gene_df.loc[gene_df['InferenceGeneSymbol'].isin(self.top_n_genes)]

        evidence_df = self.input_df.loc[self.input_df['DirectEvidence'].notnull()][['ChemicalName', 'DiseaseName', 'DirectEvidence', 'DiseaseID']]

        if self.use_gene_inference_score:
            gb_df = gene_df.pivot_table(index=['DiseaseName', 'ChemicalName', 'DiseaseID'],
                              columns='InferenceGeneSymbol',
                              values='InferenceScore',
                              aggfunc='max',
                              fill_value=0).reset_index()
        else:
            dummy_df = pd.get_dummies(gene_df, prefix='', prefix_sep='',columns=['InferenceGeneSymbol'])

            gb_df = dummy_df.groupby(['DiseaseName', 'ChemicalName', 'DiseaseID']).agg({np.max}).reset_index()
            gb_df.columns = gb_df.columns.droplevel(1)


        merged_df = gb_df.merge(evidence_df, on=['ChemicalName', 'DiseaseName', 'DiseaseID'])

        # merged_df['label'] = np.where(merged_df['DirectEvidence'] == 'marker/mechanism',
        #                  merged_df['InferenceScore'] * -1,
        #                  merged_df['InferenceScore'])


        return merged_df

    def plot_results(self, history, predicted_values, y_test, accuracy) -> float:
        """
        Plot the results of the model's performance.

        Args:
            history: Training history of the model.
            predicted_values: Predicted values from the model.
            y_test: True labels of the test data.
            accuracy: Accuracy of the model.

        Returns:
            AUC score

        Raises:
            NotImplementedError: If `classification` is 'categorical'.

        """
        if self.classification == 'categorical':
            raise NotImplementedError("Subclass categorical classifications and override this method.")

        auc_score = roc_auc_score(y_test, predicted_values)

        if not self.show_plots:
            return auc_score

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        # Plot accuracy
        axes[0][0].plot(history.history['accuracy'], label='accuracy')
        axes[0][0].plot(history.history['val_accuracy'], label='val_accuracy')
        axes[0][0].text(2, history.history['accuracy'][0] + .005, 'accuracy: {:.4f}'.format(accuracy))
        axes[0][0].legend()

        # Plot loss
        axes[0][1].plot(history.history['loss'], label='loss')
        axes[0][1].plot(history.history['val_loss'], label='val_loss')
        axes[0][1].legend()

        fig.tight_layout()

        fpr, tpr, thresholds = roc_curve(y_test, predicted_values)

        # Plot ROC curve
        axes[1][0].plot(fpr, tpr)
        axes[1][0].text(0.7, 0.9, 'auc: {:.4f}'.format(auc_score))
        axes[1][0].axis([-.05, 1.1, 0, 1.05])

        # Plot confusion matrix
        cm = confusion_matrix(y_test, np.where(predicted_values > 0.5, 1, 0))
        labels = ["Non Target", "Target"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=axes[1][1])

        return auc_score

    def get_class_weights(self, labels: pd.Series) -> Dict:
        """
        Determine the weights to assign to each class based on the distribution of classes.

        Args:
            labels: A pandas Series containing the class labels.

        Returns:
            A dictionary containing the weights assigned to each class.

        """
        classes = labels.unique()
        total_classes = len(classes)
        total = len(labels)

        class_weights = {}
        for cl in classes:
            count_of_this_class = len(labels[labels == cl])
            class_weights[cl] = (1 / count_of_this_class) * (total / total_classes)

        return class_weights

    def get_model(self, input_shape: int, output_shape: int) -> Sequential:
        """
        Create and compile a neural network model.

        Args:
            input_shape: The input shape of the model.
            output_shape: The output shape of the model.

        Returns:
            The compiled neural network model.

        """
        model = Sequential()

        model.add(Dense(60, input_dim=input_shape, activation='relu'))
        model.add(Dense(6, input_dim=input_shape, activation='relu'))
        model.add(Dense(output_shape, activation='sigmoid'))

        model.compile(loss=self.classification + '_crossentropy', optimizer='adam', metrics=METRICS)

        return model

    def over_sample(self, X_train: pd.DataFrame, y_train: np.ndarray, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform oversampling on the training data to balance the classes.

        Args:
            X_train: The training features.
            y_train: The training labels.

        Returns:
            A tuple containing the oversampled features and labels.

        Raises:
            Exception: If the classification type is 'categorical'.

        """

        if self.classification == 'categorical':
            raise Exception('Oversampling is not supported with categorical classifications.')

        return SMOTE(k_neighbors=k_neighbors).fit_resample(X_train, y_train)
    def train_model(self, train_df: pd.DataFrame) -> Tuple[List[dict], Sequential, float, Dict[str, float]]:
        """Train a model using the provided training dataframe.

        Args:
            train_df: The training dataframe.

        Returns:
            A tuple containing the model history, trained model, AUC score, and dictionary of metrics.

        """

        gene_columns = train_df.columns.intersection(self.top_n_genes)
        shuffled_df = train_df.sample(frac=1)
        features, labels = shuffled_df[gene_columns], shuffled_df['binary_label']

        if self.classification == 'categorical':
            enc = OneHotEncoder()
            labels = enc.fit_transform(shuffled_df['categorical_label'].values[:, np.newaxis]).toarray()

        output_layer_num = labels.shape[1] if self.classification == 'categorical' else 1
        model = self.get_model(features.shape[1], output_layer_num)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=0, train_size=TRAIN_SIZE)

        if self.oversample:
            X_train, y_train = self.over_sample(X_train, y_train, 5)

        callbacks = []
        if self.stop_early:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2))

        model_fit_kw = {
            'x': X_train,
            'y': y_train,
            'epochs': 25,
            'validation_split': 0.2,
            'callbacks': callbacks,
            'verbose': VERBOSE
        }
        if self.use_class_weights:
            label_column = self.classification + '_label'
            model_fit_kw['class_weight'] = self.get_class_weights(train_df[label_column])

        # Fit the model
        history = model.fit(**model_fit_kw)

        # Make predictions on the test set
        predicted_values = model.predict(X_test)

        # Evaluate model metrics
        model_metrics = model.evaluate(X_test, y_test, verbose=VERBOSE)
        model_keys = ['loss'] + [m.name for m in METRICS]
        metrics_info = dict(zip(model_keys, model_metrics))

        auc = self.plot_results(history, predicted_values, y_test, metrics_info.get('accuracy'))

        return history, model, auc, metrics_info

    def apply_category(self, row: pd.Series) -> int:
        """Apply a category label based on the values in the input row.

        Args:
            row: A pandas Series representing a single row of data.

        Returns:
            An integer representing the category label.

        """
        if row.binary_label == 0:
            return 0  # 'Not Relevant'
        if row.DirectEvidence == 'marker/mechanism':
            return 1  # 'Negative'
        else:
            return 2  # 'Therapeutic'

    def set_label(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Set labels in the training DataFrame based on either a binary or
        categorical label.  We are tryign to predict association for a set
        of diseases based on a gene netweork so we set the labels if the
        disease in the observation is part of our target diseases.

        Args:
            train_df: Training DataFrame.

        Returns:
            Training DataFrame with labels.
        """
        train_df['binary_label'] = np.where(train_df['DiseaseID'].isin(self.target_diseases), 1, 0)

        if self.classification == 'categorical':
            train_df['categorical_label'] = train_df.apply(lambda row: self.apply_category(row), axis=1)

        return train_df

    def main(self) -> Tuple[Dict[str, Any], Sequential]:
        """Run the main process.

        Returns:
            A tuple containing a dictionary of model metrics and the trained model.

        """
        train_df = self.prep_training_data()
        train_df = self.set_label(train_df)
        history, model, auc, model_metrics = self.train_model(train_df)

        model_metrics['parent_disease'] = self.parent_disease
        model_metrics['gene_count'] = self.gene_count
        model_metrics['show_plots'] = self.show_plots
        model_metrics['use_class_weights'] = self.use_class_weights
        model_metrics['oversample'] = self.oversample
        model_metrics['model_type'] = self.model_type

        return model_metrics, model
import os
import gzip
import shutil
import warnings
import requests
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import eli5

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ## TO DO set in docker
import tensorflow as tf
from keras import Sequential, metrics
from keras.layers import Input, Dense, BatchNormalization, LSTM, Embedding, Bidirectional, Normalization, Conv1D, Dropout, MaxPool2D, MaxPooling1D, Flatten
from keras.models import Model

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ctd.utils import get_disease_hierarchy


VERBOSE=0
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
        model_type: str = 'DNN',
        use_gene_inference_score: bool = False,
        show_feature_importance: bool = False
    ) -> None:
        """
        Initialize the DiseaseClassifier class by getting the target dieseases
        based on the "parent disease" that is passed as an input. For example
        passing in neurodegenerative disease will get Parkinson Alzheiemers and
        so an.  This gives the algorithm significantly more positive labels to
        train against.

        The algorithm also accepts "gene count" which will determine the number
        of featuers to used in the classifier based on the N top most frequently
        seen genes in the dataset.

        Args:
            input_df: The input dataframe containing the raw
            data from the Chemical -> Gene -> Disease data - use
            ctd.get_data('ChemicalDiseaseInteractions') to pull this
            parent_disease: The parent disease for which classification is
            performed.  All children of this disease are also market as positive
            labels.
            gene_count: The number of genes ( or features ) to consider
            for classification based on frequency in the dataset
            show_plots: Flag indicating whether to show plots or not.
            use_class_weights: Flag indicating whether to use class weights.
            oversample: Flag indicating whether to perform oversampling.
            classification: The type of classification, default is 'binary' but
            'categorical' can be used as well.
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
        self.model_type = model_type
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
        observation.

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

        return merged_df

    def plot_results(self, history, predicted_values, y_test, accuracy) -> float:
        """
        Plot results of
            -  loss / accuracy over the epochs performed
            -  ROC Curve if it's a binary classifier
            -  Confusion Matrix

        Args:
            history: Training history of the model.
            predicted_values: Predicted values from the model.
            y_test: True labels of the test data.
            accuracy: Accuracy of the model.

        Returns:
            AUC score or Zero if classification = categorical

        """
        if not self.show_plots:
            return 0

        if self.classification == 'categorical':
            cm = confusion_matrix(y_test.argmax(axis=1), predicted_values.argmax(axis=1))

            labels = ['Not Relevant', 'Therapeutic', 'Negative']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

            disp.plot(cmap=plt.cm.Blues)
            return 0
            # raise NotImplementedError("Subclass categorical classifications and override this method.")

        auc_score = roc_auc_score(y_test, predicted_values)

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
        Determine the weights to assign to each class based on the distribution\
        of classes.

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
        Create and compile a model.  If using a DNN as the model_type
        we use a simple two layer network model.  Otherwise we use a linear
        model.

        We compile the loss function as either binary_crossentropy or
        categorical_crossentropy and use the metrics defined at the topic of
        the file -- these are the sample metrics that are return in the
        model_metrics object from the main class of the algorithm.

        Args:
            input_shape: The input shape of the model - the number of features
            output_shape: The output shape of the model - i.e. 1 if a binary
            classifier else, the length of the target class.
        Returns:
            The compiled neural network model.

        """
        model = Sequential()

        if self.model_type == 'DNN':
            model.add(Dense(60, input_dim=input_shape, activation='relu'))
            model.add(Dense(6, input_dim=input_shape, activation='relu'))
            model.add(Dense(output_shape, activation='sigmoid'))
        elif self.model_type == 'LINEAR':
            model.add(Dense(output_shape, activation='linear', input_shape=(input_shape,)))

        model.compile(loss=self.classification + '_crossentropy', optimizer='adam', metrics=METRICS)

        return model

    def over_sample(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            raise Exception('over sampling not supported with categorical classifications')
        bool_train_labels = y_train != 0

        pos_features = X_train[bool_train_labels]
        neg_features = X_train[~bool_train_labels]

        pos_labels = y_train[bool_train_labels]
        neg_labels = y_train[~bool_train_labels]

        ids = np.arange(len(pos_features))
        choices = np.random.choice(ids, len(neg_features))

        res_pos_features = pos_features.iloc[choices, :]
        res_pos_labels = pos_labels.values[choices] # pos_labels.array(choices)

        res_pos_features.shape

        resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
        resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

        order = np.arange(len(resampled_labels))
        np.random.shuffle(order)
        resampled_features = resampled_features[order]
        resampled_labels = resampled_labels[order]

        return resampled_features, resampled_labels

    def train_model(self, train_df: pd.DataFrame) -> Tuple[List[dict], Sequential, float, Dict[str, float]]:
        """Train a model using the provided training dataframe.

        If usign a categorical classification, we onehot encode each class
        as it's own binary label.

        We then get the model ( DNN or Linear ) get the train test split and
        apply some sort of oversampling technique.

        Finally, plot the results of the model run to understand the performance
        over each epoch, the AUC and the confusion matrix.

        Args:
            train_df: The training dataframe.

        Returns:
            A tuple containing:
            - The model history
            - Trained model
            - AUC score
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
            X_train, y_train = self.over_sample(X_train, y_train)

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

        if self.model_type == 'LINEAR':
            weights = model.get_weights()[0]
            weights = np.abs(weights)
            metrics_info['feature_importance'] = dict(zip(X_train.columns, weights))

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
        """Run the main process which preps the data, sets labels and trains the model
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


if __name__ == '__main__':
    # from src.disease_classifier import DiseaseClassifier
    from disease_classifier import DiseaseClassifier
    from ctd import get_data
    input_df = get_data('ChemicalDiseaseInteractions')

    kw = {
        'input_df': input_df,
        'parent_disease': 'MESH:D019636', # neurodegenerative diseases
        'gene_count': 10,
        'show_plots': True,
        'use_class_weights': False,
        'oversample': False,
        'show_feature_importance': True
    }
    dc = DiseaseClassifier(**kw)
    dc_mm, model = dc.main()

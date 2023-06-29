import os
import unittest
import pandas as pd
import numpy as np

from mock import patch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import Sequential
from keras.layers import Dense
from imblearn.over_sampling import SMOTE

import src.disease_classifier
from src.disease_classifier import DiseaseClassifier

class DiseaseClassifierTest(unittest.TestCase):
    @patch('src.disease_classifier.DiseaseClassifier.get_genes')
    def setUp(self, mock_get_genes):
        mock_get_genes.return_value = ['GeneA', 'GeneB', 'GeneC']
        # input_df = pd.read_csv('src/test_data/test_df.csv')  # Provide sample data here

        # Set up the parameters for initialization
        parent_disease = 'X'
        gene_count = 100
        show_plots = False
        use_class_weights = True
        oversample = False

        # Initialize the DiseaseClassifier
        self.classifier = DiseaseClassifier(
            pd.DataFrame(),
            parent_disease,
            gene_count,
            show_plots,
            use_class_weights,
            oversample
        )
    def test_disease_classifier_initialization(self):
        # Create a sample input DataFrame
        input_df = pd.read_csv('src/test_data/test_df.csv')  # Provide sample data here

        # Set up the parameters for initialization
        parent_disease = 'MESH:D019636'
        gene_count = 100
        show_plots = False
        use_class_weights = True
        oversample = False

        # Initialize the DiseaseClassifier
        classifier = DiseaseClassifier(
            input_df,
            parent_disease,
            gene_count,
            show_plots,
            use_class_weights,
            oversample
        )

        # Assert that the attributes are set correctly
        pd.testing.assert_frame_equal(classifier.input_df, input_df)
        self.assertEqual(classifier.parent_disease, parent_disease)
        self.assertEqual(classifier.gene_count, gene_count)
        self.assertEqual(classifier.show_plots, show_plots)
        self.assertEqual(classifier.use_class_weights, use_class_weights)
        self.assertEqual(classifier.oversample, oversample)
        self.assertEqual(classifier.classification, 'binary')
        self.assertEqual(classifier.model_type, 'DNN')

    def test_disease_classifier_exception(self):
        # Create a sample input DataFrame
        input_df = pd.DataFrame()  # Provide sample data here

        # Set up the parameters for initialization with conflicting flags
        parent_disease = 'MESH:D019636'
        gene_count = 100
        show_plots = False
        use_class_weights = True
        oversample = True

        # Assert that an exception is raised when both use_class_weights and oversample are True
        with self.assertRaises(Exception):
            DiseaseClassifier(
                input_df,
                parent_disease,
                gene_count,
                show_plots,
                use_class_weights,
                oversample
            )

    def test_get_genes(self):
        # Create a sample input DataFrame
        input_data = {
            'InferenceGeneSymbol': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
        }
        input_df = pd.DataFrame(input_data)

        # Create an instance of DiseaseClassifier
        classifier = DiseaseClassifier(input_df, parent_disease='MESH:D019636', gene_count=3,
                                       show_plots=False, use_class_weights=False, oversample=False)

        # Call the get_genes method
        result_genes = classifier.get_genes()

        # Assert the expected output
        expected_genes = ['GeneA', 'GeneB', 'GeneC']
        self.assertListEqual(result_genes.tolist(), expected_genes)

    def test_prep_training_data(self):
        # Create a sample input DataFrame

        input_df = pd.read_csv('src/test_data/test_df.csv')

        # Create an instance of DiseaseClassifier
        classifier = DiseaseClassifier(input_df, parent_disease='MESH:D019636', gene_count=10,
                                       show_plots=False, use_class_weights=False, oversample=False)


        result_df = classifier.prep_training_data()
        result_df.to_csv('src/test_data/one_hot_df.csv', index=False)

        expected_df = pd.read_csv('src/test_data/one_hot_df.csv')

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_prep_training_data(self):
        '''
        This goes through how the data is pivoted to create features from
        each gene interaction.  This uses the onehot encoding technique to add
        binary labels, while the test below uses the inference scores as
        continuous feautre values
        '''
        input_data = {
            'ChemicalName': ['Chem1', 'Chem1', 'Chem1', 'Chem1'],
            'DiseaseName': ['Disease1', 'Disease1', 'Disease1', 'Disease1'],
            'InferenceGeneSymbol': ['GeneA', 'GeneB', 'GeneC', None],
            'InferenceScore': [.1, .1, .1, None],
            'DiseaseID': ['ID1', 'ID1', 'ID1', 'ID1'],
            'DirectEvidence': [None, None, None, 'marker/mechanism']
        }
        input_df = pd.DataFrame(input_data)
        # Create an instance of DiseaseClassifier
        classifier = DiseaseClassifier(input_df, parent_disease='MESH:D019636', gene_count=50,
                                       show_plots=False, use_class_weights=False, oversample=False)

        # Call the prep_training_data method
        result_df = classifier.prep_training_data()

        # Assert the expected output
        expected_data = {
            'DiseaseName': ['Disease1'],
            'ChemicalName': ['Chem1'],
            'DiseaseID': ['ID1'],
            'InferenceScore':[0.1],
            'GeneA': [True],
            'GeneB': [True],
            'GeneC': [True],
            'DirectEvidence': ['marker/mechanism']
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_prep_training_data_inference_score(self):
        # Create a sample input DataFrame
        input_data = {
            'ChemicalName': ['Chem1', 'Chem1', 'Chem1', 'Chem1'],
            'DiseaseName': ['Disease1', 'Disease1', 'Disease1', 'Disease1'],
            'InferenceGeneSymbol': ['GeneA', 'GeneB', 'GeneC', None],
            'InferenceScore': [.4, .2, .9, None],
            'DiseaseID': ['ID1', 'ID1', 'ID1', 'ID1'],
            'DirectEvidence': [None, None, None, 'marker/mechanism']
        }

        input_df = pd.DataFrame(input_data)
        # Create an instance of DiseaseClassifier
        classifier = DiseaseClassifier(input_df, parent_disease='MESH:D019636', gene_count=50,
                                       show_plots=False, use_class_weights=False, oversample=False, use_gene_inference_score=True)

        # Call the prep_training_data method
        result_df = classifier.prep_training_data()

        # Assert the expected output
        expected_data = {
            'DiseaseName': ['Disease1'],
            'ChemicalName': ['Chem1'],
            'DiseaseID': ['ID1'],
            'GeneA': [.4],
            'GeneB': [.2],
            'GeneC': [.9],
            'DirectEvidence': ['marker/mechanism'],

        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df, expected_df)

    @patch('src.disease_classifier.DiseaseClassifier.get_genes')
    def test_get_class_weights(self, mock_get_genes):
        mock_get_genes.return_value = ['GeneA', 'GeneB', 'GeneC']

        classifier = DiseaseClassifier(pd.DataFrame(), parent_disease='MESH:D019636', gene_count=50,
                                       show_plots=False, use_class_weights=False, oversample=False)

        # Mock data for testing
        labels = pd.Series([0, 1, 0, 1, 1, 1])

        # Call the method
        class_weights = classifier.get_class_weights(labels)

        # Expected class weights
        expected_weights = {0: 1.5, 1: 0.75}

        # Assert the returned class weights
        self.assertEqual(class_weights, expected_weights)

    @patch('src.disease_classifier.DiseaseClassifier.get_genes')
    def test_get_model(self, mock_get_genes):
        mock_get_genes.return_value = ['GeneA', 'GeneB', 'GeneC']
        classifier = DiseaseClassifier(pd.DataFrame(), parent_disease='MESH:D019636', gene_count=50,
                               show_plots=False, use_class_weights=False, oversample=False)

        classifier.classification = 'binary'  # Set the classification type

        # Mock data for testing
        input_shape = 10
        output_shape = 1

        # Call the method
        model = classifier.get_model(input_shape, output_shape)

        # Assert the model type
        self.assertIsInstance(model, Sequential)

        # Assert the number of layers in the model
        self.assertEqual(len(model.layers), 3)

        # Assert the type of the layers
        self.assertIsInstance(model.layers[0], Dense)
        self.assertIsInstance(model.layers[1], Dense)
        self.assertIsInstance(model.layers[2], Dense)

        # Assert the activation functions of the layers
        self.assertEqual(model.layers[0].activation.__name__, 'relu')
        self.assertEqual(model.layers[1].activation.__name__, 'relu')
        self.assertEqual(model.layers[2].activation.__name__, 'sigmoid')

    @patch('src.disease_classifier.DiseaseClassifier.get_genes')
    def test_train_model(self, mock_get_genes):
        mock_get_genes.return_value = ['GeneA', 'GeneB', 'GeneC']

        input_df = pd.DataFrame({
            'DiseaseID': ['D001', 'D002', 'D001', 'D001'],
            'ChemicalName': ['Chem1', 'Chem2', 'Chem3', 'Chem4'],
            'binary_label': [1,1,1,0],
        })
        classifier = DiseaseClassifier(input_df, parent_disease='MESH:D019636', gene_count=50,
                               show_plots=False, use_class_weights=False, oversample=False)
        classifier.classification = 'binary'
        # Call the train_model method
        history, model, auc, metrics_info = classifier.train_model(input_df)

        # Example assertions to check if the model and history are not None
        self.assertIsNotNone(history)
        self.assertIsNotNone(model)

        self.assertIsInstance(metrics_info, dict)
        self.assertIn('loss', metrics_info)
        self.assertIn('accuracy', metrics_info)
    @patch('src.disease_classifier.DiseaseClassifier.get_genes')
    def test_apply_category(self, mock_get_genes):
        self.assertEqual(self.classifier.apply_category(pd.Series({'binary_label': 0, 'DirectEvidence': 'marker/mechanism'})), 0)
        self.assertEqual(self.classifier.apply_category(pd.Series({'binary_label': None, 'DirectEvidence': 'marker/mechanism'})), 1)
        self.assertEqual(self.classifier.apply_category(pd.Series({'binary_label': None, 'DirectEvidence': 'therapeutic'})), 2)
    @patch('src.disease_classifier.DiseaseClassifier.get_genes')
    def test_set_label(self, mock_get_genes):

        input_df = pd.DataFrame({
            'DiseaseID': ['D001', 'D002', 'D001', 'D001'],
            'ChemicalName': ['Chem1', 'Chem2', 'Chem3', 'Chem4']
        })
        target_df = pd.DataFrame({
            'DiseaseID': ['D001', 'D002', 'D001', 'D001'],
            'ChemicalName': ['Chem1', 'Chem2', 'Chem3', 'Chem4'],
            'binary_label': [0, 1, 0, 0],
        })
        parent_disease = 'MESH:D019636'
        gene_count = 100
        show_plots = True
        use_class_weights = False
        oversample = True
        classification = 'binary'

        classifier = DiseaseClassifier(input_df, parent_disease, gene_count, show_plots, use_class_weights, oversample,
                                       classification)
        classifier.target_diseases = ['D002']
        labeled_df = classifier.set_label(input_df)

        # since we're predicting 'D002' that should be the
        # only disease where binary_label = True
        pd.testing.assert_frame_equal(labeled_df, target_df)

if __name__ == '__main__':
    unittest.main()

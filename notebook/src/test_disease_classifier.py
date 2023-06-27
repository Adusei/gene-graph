import unittest
import pandas as pd

from disease_classifier import DiseaseClassifier


class DiseaseClassifierTest(unittest.TestCase):
    # def test_disease_classifier_initialization(self):
    #     # Create a sample input DataFrame
    #     input_df = pd.read_csv('src/test_data/test_df.csv')  # Provide sample data here
    #
    #     # Set up the parameters for initialization
    #     parent_disease = 'MESH:D019636'
    #     gene_count = 100
    #     show_plots = False
    #     use_class_weights = True
    #     oversample = False
    #
    #     # Initialize the DiseaseClassifier
    #     classifier = DiseaseClassifier(
    #         input_df,
    #         parent_disease,
    #         gene_count,
    #         show_plots,
    #         use_class_weights,
    #         oversample
    #     )
    #
    #     # Assert that the attributes are set correctly
    #     pd.testing.assert_frame_equal(classifier.input_df, input_df)
    #     self.assertEqual(classifier.parent_disease, parent_disease)
    #     self.assertEqual(classifier.gene_count, gene_count)
    #     self.assertEqual(classifier.show_plots, show_plots)
    #     self.assertEqual(classifier.use_class_weights, use_class_weights)
    #     self.assertEqual(classifier.oversample, oversample)
    #     self.assertEqual(classifier.classification, 'binary')
    #     self.assertEqual(classifier.model_type, 'DNN')
    #
    # def test_disease_classifier_exception(self):
    #     # Create a sample input DataFrame
    #     input_df = pd.DataFrame()  # Provide sample data here
    #
    #     # Set up the parameters for initialization with conflicting flags
    #     parent_disease = 'MESH:D019636'
    #     gene_count = 100
    #     show_plots = False
    #     use_class_weights = True
    #     oversample = True
    #
    #     # Assert that an exception is raised when both use_class_weights and oversample are True
    #     with self.assertRaises(Exception):
    #         DiseaseClassifier(
    #             input_df,
    #             parent_disease,
    #             gene_count,
    #             show_plots,
    #             use_class_weights,
    #             oversample
    #         )
    #
    # def test_get_genes(self):
    #     # Create a sample input DataFrame
    #     input_data = {
    #         'InferenceGeneSymbol': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
    #     }
    #     input_df = pd.DataFrame(input_data)
    #
    #     # Create an instance of DiseaseClassifier
    #     classifier = DiseaseClassifier(input_df, parent_disease='MESH:D019636', gene_count=3,
    #                                    show_plots=False, use_class_weights=True, oversample=False)
    #
    #     # Call the get_genes method
    #     result_genes = classifier.get_genes()
    #
    #     # Assert the expected output
    #     expected_genes = ['GeneA', 'GeneB', 'GeneC']
    #     self.assertListEqual(result_genes.tolist(), expected_genes)
    #
    # def test_prep_training_data(self):
    #     # Create a sample input DataFrame
    #
    #     input_df = pd.read_csv('src/test_data/test_df.csv')
    #
    #     # Create an instance of DiseaseClassifier
    #     classifier = DiseaseClassifier(input_df, parent_disease='MESH:D019636', gene_count=10,
    #                                    show_plots=False, use_class_weights=True, oversample=False)
    #
    #
    #     result_df = classifier.prep_training_data()
    #     result_df.to_csv('src/test_data/one_hot_df.csv', index=False)
    #
    #     expected_df = pd.read_csv('src/test_data/one_hot_df.csv')
    #
    #     pd.testing.assert_frame_equal(result_df, expected_df)
    #
    # def plot_results(self, history: tf.keras.callbacks.History, predicted_values: np.ndarray, y_test: np.ndarray,
    #                  accuracy: float) -> Union[float, None]:
    #     """
    #     Plot the results of model training and evaluation.
    #
    #     Args:
    #         history: History object containing training history.
    #         predicted_values: Predicted values from the model.
    #         y_test: True labels from the test data.
    #         accuracy: Model accuracy.
    #
    #     Returns:
    #         AUC score if show_plots is False, otherwise None.
    #     """
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
                                       show_plots=False, use_class_weights=True, oversample=False)

        # Call the prep_training_data method
        result_df = classifier.prep_training_data(one_hot=True)

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
                                       show_plots=False, use_class_weights=True, oversample=False)

        # Call the prep_training_data method
        result_df = classifier.prep_training_data(one_hot=False)

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

        print('---result_df---')
        print(result_df)

        print('---expected_df---')
        print(expected_df)

        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main()

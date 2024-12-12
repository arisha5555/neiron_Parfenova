import unittest
from unittest.mock import patch, Mock
from parfenova_network import Neuron


class Test_Activation(unittest.TestCase):
    def test_activation():

        expected_result = '-'

        result = Neuron._get_activation_function('sigmoid')
        print(result)
        self.assertEqual(result, expected_result)



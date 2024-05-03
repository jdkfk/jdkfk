import unittest
from jdkfk.data.data import Dataset
import pandas as pd


class TestDataset(unittest.TestCase):

    def setUp(self):
        df = pd.read_csv('insurance.csv')
        self.dataset = Dataset(df)
    

    def test_Dataset(self):

        self.assertFalse(self.dataset.df.empty)
        print(self.dataset.df.sort_values(by='age',ascending=False).tail(10))

import unittest
import test_class
import numpy as np

test_features, test_targets, test_objects, test_probs = test_class.create_test_set()


class TestFeatureMethods(unittest.TestCase):

    def test_instance(self):
        for object in test_objects:
            self.assertIsInstance(object, test_class.Mammogram)

    def test_mean(self):
        i=0
        for row in test_features:
            self.assertTrue(1>= row[0] >= 0)
            m = test_objects[i].array.shape[0]
            n = test_objects[i].array.shape[1]
            self.assertAlmostEqual(1/(m*n)*np.sum(test_objects[i].array), row[0])
            i+=1

    def test_stddev(self):
        i=0
        for row in test_features:
            self.assertTrue(row[0] > row[1])
            m = test_objects[i].array.shape[0]
            n = test_objects[i].array.shape[1]
            self.assertAlmostEqual(np.sqrt(1/(m*n)*np.sum((test_objects[i].array - row[0])**2)), row[1])
            i+=1

    def test_smooth(self):
        for row in test_features:
            self.assertEqual(1-(1/(1+(row[1]**2))), row[2])

    def test_entropy(self):
        i = 0
        for row in test_features:
            self.assertAlmostEqual(row[3], -np.sum(test_probs[i] * np.log2(test_probs[i]), axis=0))
            i+=1

    def test_skew(self):
        i = 0
        for row in test_features:
            m = test_objects[i].array.shape[0]
            n = test_objects[i].array.shape[1]
            self.assertAlmostEqual((1/(m*n)*np.sum(((test_objects[i].array - row[0])/row[1])**3)), row[4])
            i+=1

    def test_kurt(self):
        i = 0
        for row in test_features:
            m = test_objects[i].array.shape[0]
            n = test_objects[i].array.shape[1]
            self.assertAlmostEqual((1/(m*n)*np.sum(((test_objects[i].array - row[0])/row[1])**4)) - 3, row[5])
            i+=1

    def test_uniformity(self):
        i=0
        for row in test_features:
            self.assertEqual(np.sum(test_probs[i]**2), row[6])
            i+=1


if __name__ == '__main__':
    unittest.main()
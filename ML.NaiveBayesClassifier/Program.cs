using ML.Common;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.NaiveBayesClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] contents = File.ReadAllLines("house-votes-84.data");
            var random = new Random();
            var randomOrderedContents = contents.OrderBy(x => random.Next()).ToArray();
            var accuracies = DataProcessing.TenFoldCrossValidation(randomOrderedContents, new BayesClassifierModel());
            accuracies.ForEach(curAccuracy =>
            {
                Console.WriteLine($"Chunk accuracy: {curAccuracy:P}");
            });

            Console.WriteLine($"Average accuracy for model: {accuracies.Average():P}");
            Console.ReadKey();
        }
    }
}

using ML.Common;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.DecisionTree
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] contents = File.ReadAllLines("breast-cancer.data");
            var sanitizedContents = DataProcessing.RemoveRowsWithMissingData(contents);
            var random = new Random();

            var randomOrderedContents = sanitizedContents.OrderBy(x => random.Next()).ToArray();
            var accuracies = DataProcessing.TenFoldCrossValidation(randomOrderedContents, new DecisionTreeModel());
            accuracies.ForEach(curAccuracy =>
            {
                Console.WriteLine($"Chunk accuracy: {curAccuracy:P}");
            });
            Console.WriteLine($"Average accuracy for model: {accuracies.Average():P}");
            Console.ReadKey();

        }
    }
}

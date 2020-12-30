using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Common
{
    public static class DataProcessing
    {
        public static string[] RemoveRowsWithMissingData(string[] contents)
        {
            return contents.Where(content => !content.Contains("?")).ToArray();
        }

        public static List<double> TenFoldCrossValidation<T>(string[] contents, IModel<T> model, int classAttrPosition = 0)
            where T : Enum
        {
            var accuracies = new List<double>();
            for (int i = 0; i < 10; i++)
            {
                var aTenth = contents.Length / 10;
                var validatingSet = contents.Skip(i * aTenth).Take(aTenth).ToList();
                var trainingSet = contents.Take(i * aTenth).ToList();
                trainingSet.AddRange(contents.Skip(i * aTenth + aTenth));
                model.TrainModel(trainingSet);
                var curAccuracy = CalculateAccuracy(model, validatingSet, classAttrPosition);
                accuracies.Add(curAccuracy);
            }
            return accuracies;
        }

        public static double CalculateAccuracy<T>(IModel<T> model, List<string> validatingSet, int classPos)
            where T : Enum
        {
            var predictedCnt = 0;
            for (int i = 0; i < validatingSet.Count; i++)
            {
                var curRow = validatingSet[i].Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                var actualClass = ReflectionHelpers.EnumGetValueFromDescription<T>(curRow[classPos]);
                var predictedClass = model.Classify(curRow.Skip(1).ToArray());
                if (actualClass.Equals(predictedClass))
                {
                    predictedCnt++;
                }
            }
            return (double)predictedCnt / validatingSet.Count;
        }

        
    }
}

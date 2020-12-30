using ML.Common;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.NaiveBayesClassifier
{
    enum Class
    {
        Republican,
        Democrat
    }

    class BayesClassifierModel : IModel<Class>
    {
        public double RepublicanProb { get; set; }

        public double DemocratProb { get; set; }

        public double[][] ProbYes { get; set; }

        public double[][] ProbNo { get; set; }

        public double[][] ProbMissing { get; set; }

        public void TrainModel(List<string> contents)
        {
            Dictionary<Class, List<string[]>> valuesByClass = contents
                .Select(content => content.Split(',', StringSplitOptions.RemoveEmptyEntries))
                .GroupBy(row => row[0] == "republican" ? Class.Republican : Class.Democrat)
                .ToDictionary(group => group.Key, group => group.ToList());

            this.RepublicanProb = (double)valuesByClass[Class.Republican].Count / contents.Count();
            this.DemocratProb = (double)valuesByClass[Class.Democrat].Count / contents.Count();
            this.ProbYes = new double[2][]
            {
                new double[16],
                new double[16],
            };
            this.ProbNo = new double[2][]
            {
                new double[16],
                new double[16],
            };
            this.ProbMissing = new double[2][]
            {
                new double[16],
                new double[16],
            };

            for (int k = 0; k < 2; k++)
            {
                var curClass = k == 0 ? Class.Republican : Class.Democrat;
                var allCnt = k == 0 ? valuesByClass[Class.Republican].Count : valuesByClass[Class.Democrat].Count + 1;
                for (int i = 1; i < 17; i++)
                {
                    int yesByCurAttrCnt = 1;
                    int noByCurAttrCnt = 1;
                    int missingByCurAttrCnt = 1;
                    valuesByClass[curClass].ForEach(row =>
                    {
                        if (row[i] == "y")
                        {
                            yesByCurAttrCnt += 1;
                        }
                        else if (row[i] == "n")
                        {
                            noByCurAttrCnt += 1;
                        }
                        else
                        {
                            missingByCurAttrCnt += 1;
                        }
                    });

                    this.ProbYes[k][i - 1] = (double)yesByCurAttrCnt / allCnt;
                    this.ProbNo[k][i - 1] = (double)noByCurAttrCnt / allCnt;
                    this.ProbMissing[k][i - 1] = (double)missingByCurAttrCnt / allCnt;
                }
            }
        }

        public Class Classify(string[] test)
        {
            var allProbs = new List<(Class, double)>();
            for (int i = 0; i < 2; i++)
            {
                double curProb = i == 0 ? Math.Log(this.RepublicanProb) : Math.Log(this.DemocratProb);

                for (int j = 0; j < 16; j++)
                {
                    var curAttrProb = test[j] switch
                    {
                        "y" => this.ProbYes[i][j],
                        "n" => this.ProbNo[i][j],
                        _ => this.ProbMissing[i][j],
                    };
                    curProb += Math.Log(curAttrProb);
                }
                allProbs.Add((i == 0 ? Class.Republican : Class.Democrat, curProb));
            }

            return allProbs.OrderByDescending(x => x.Item2).First().Item1;
        }
    }
}

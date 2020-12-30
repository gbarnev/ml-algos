using ML.Common;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace ML.DecisionTree
{
    enum Class
    {
        [Description("recurrence-events")]
        RecurrenceEvents,

        [Description("no-recurrence-events")]
        NoRecurrenceEvents
    }

    enum BreastCancerAttribute
    {
        Class,
        Age,
        Menopause,
        TumorSize,
        InvNodes,
        NodeCaps,
        DegMalig,
        Breast,
        BreastQuad,
        Irradiat,
    }

    class DecisionTreeModel : IModel<Class>
    {
        private class Node
        {
            public Node()
            {
                this.Children = new Dictionary<string, Node>();
            }

            public BreastCancerAttribute Attribute { get; set; }

            public double Entropy { get; set; }

            public Class? DominantClass { get; set; }

            public Node Parent { get; set; }

            public Dictionary<string, Node> Children { get; set; }
        }

        public static int MinContentsToDivide = 62;
        private Node rootNode;

        public DecisionTreeModel()
        {
            this.rootNode = new Node();
        }

        public Class Classify(string[] entry)
        {
            var curNode = this.rootNode;
            while (curNode.Attribute != BreastCancerAttribute.Class)
            {
                //entry doesn't consist of a class, so substract one from the index
                var idx = (int)curNode.Attribute - 1;
                var data = entry[idx];

                if (!curNode.Children.ContainsKey(data))
                {
                    //Proceed to path with largest entropy not to confuse predictions
                    curNode = curNode.Children.Select(pair => pair.Value)
                        .OrderByDescending(node => node.Entropy)
                        .First();
                    continue;
                }

                curNode = curNode.Children[data];
            }
            return curNode.DominantClass.Value;
        }

        public void TrainModel(List<string> contents)
        {
            List<string[]> parsedContents = contents
                .Select(c => c.Split(',', StringSplitOptions.RemoveEmptyEntries))
                .ToList();
            this.BuildDecisionTree(null, null, parsedContents, ref this.rootNode);
        }

        private void BuildDecisionTree(Node parent, string edge, List<string[]> filteredContents, ref Node rootNode)
        {
            var attributesCnt = filteredContents[0].Length;
            var contentsCnt = filteredContents.Count;
            var recurrenceEventsCnt = filteredContents.Where(x => x[0] == "recurrence-events").Count();
            var noRecurrenceEventsCnt = filteredContents.Count - recurrenceEventsCnt;
            var entropy = this.Entropy((double)recurrenceEventsCnt / contentsCnt, (double)noRecurrenceEventsCnt / contentsCnt);

            if (entropy == 0 || filteredContents.Count <= MinContentsToDivide)
            {
                var leafNode = new Node()
                {
                    Attribute = BreastCancerAttribute.Class,
                    DominantClass = recurrenceEventsCnt > noRecurrenceEventsCnt ?
                        Class.RecurrenceEvents : Class.NoRecurrenceEvents,
                    Parent = parent,
                    Entropy = entropy,
                };

                if (recurrenceEventsCnt == noRecurrenceEventsCnt)
                {
                    //Find out which class dominates in the closest parent which has a dominant class
                    var curParent = parent;
                    while (curParent != null && !curParent.DominantClass.HasValue)
                    {
                        curParent = curParent.Parent;
                    }
                    leafNode.DominantClass = curParent.DominantClass.Value;
                }

                if (parent != null)
                {
                    parent.Children.Add(edge, leafNode);
                }

                return;
            }

            List<(int attrIdx, double gain)> gains = new List<(int, double)>();

            //TODO Skip grouping and calculating mutual entropies for the attributes which
            //are already present in the branch.
            for (int i = 1; i < attributesCnt; i++)
            {
                double combinedEntropy = 0;
                var byAttrProperties = filteredContents.GroupBy(content => content[i]);
                foreach (var group in byAttrProperties)
                {
                    var groupCnt = group.Count();
                    var curGroupProb = (double)groupCnt / contentsCnt;
                    var recurrenceEventsPerGroup = group.Where(row => row[0] == "recurrence-events").Count();
                    var nonRecurrenceEventsPerGroup = groupCnt - recurrenceEventsPerGroup;
                    var propertyCombinedEntropy = this.Entropy(
                        (double)recurrenceEventsPerGroup / groupCnt, (double)nonRecurrenceEventsPerGroup / groupCnt);
                    combinedEntropy += curGroupProb * propertyCombinedEntropy;
                }
                var gainForAttr = entropy - combinedEntropy;
                gains.Add((i, gainForAttr));
            }

            (var attrIdx, var gain) = gains.OrderByDescending(x => x.gain).First();
            var newNode = new Node()
            {
                Attribute = (BreastCancerAttribute)attrIdx,
                Parent = parent,
                Entropy = entropy,
            };

            if (recurrenceEventsCnt != noRecurrenceEventsCnt)
            {
                newNode.DominantClass = recurrenceEventsCnt > noRecurrenceEventsCnt ?
                    Class.RecurrenceEvents : Class.NoRecurrenceEvents;
            }

            if (parent != null)
            {
                parent.Children.Add(edge, newNode);
            }
            else
            {
                rootNode = newNode;
            }

            var groupsByAttr = filteredContents.GroupBy(row => row[attrIdx]);
            foreach (var group in groupsByAttr)
            {
                this.BuildDecisionTree(newNode, group.Key, group.ToList(), ref rootNode);
            }
        }

        private double Entropy(double x1, double x2)
        {
            var entropy = -x1 * Math.Log2(x1) - x2 * Math.Log2(x2);
            if (double.IsNaN(entropy))
            {
                return 0;
            }
            return entropy;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Common
{
    public interface IModel<TClass> where TClass : Enum
    {
        void TrainModel(List<string> contents);
        
        TClass Classify(string[] test);
    }
}

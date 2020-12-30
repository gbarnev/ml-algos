using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace ML.Common
{
    public static class ReflectionHelpers
    {
        public static T EnumGetValueFromDescription<T>(string description) where T : Enum
        {
            foreach (var field in typeof(T).GetFields())
            {
                if (Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute))
                    is DescriptionAttribute attribute)
                {
                    if (attribute.Description.Equals(description, StringComparison.OrdinalIgnoreCase))
                        return (T)field.GetValue(null);
                }
                else
                {
                    if (field.Name.Equals(description, StringComparison.OrdinalIgnoreCase))
                        return (T)field.GetValue(null);
                }
            }

            throw new ArgumentException("Not found", nameof(description));
        }
    }
}

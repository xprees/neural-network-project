namespace DataLoading;

public class Evaluator(float maxError = 0.0001f)
{

    public bool EvaluateCompleteResult(float[] predicted, float[] actual)
    {
        return predicted.
            Select((value, index) => MathF.Abs(value - actual[index]) <= maxError)
            .All(condition => condition);
    }

    public int[] ConvertToClasses(float[][] toConvert)
    {
        return toConvert.Select(value => Array.IndexOf(value, value.Max())).ToArray();
    }

    public bool[] EvaluateCompletedResults(float[][] predicted, float[][] actual)
    {
        return predicted.Select((value, index) => EvaluateCompleteResult(value, actual[index])).ToArray();
    }

    public StatisticalMetrics EvaluateModel(float[][] predicted, float[][] actual)
    {
        if (predicted.Length != actual.Length)
        {
            throw new ArgumentException("Predicted and actual must have the same length");
        }
        var statisticalMetrics = new StatisticalMetrics();
        var convertedPredicted = ConvertToClasses(predicted);
        var convertedActual = ConvertToClasses(actual);
        for (int classIndex = 0; classIndex < 10; classIndex++) // Not the most effective, can be remade later
        {
            int fp = 0;
            int fn = 0;
            int tp = 0;
            int tn = 0;

            for (int arrayIndex = 0; arrayIndex < convertedPredicted.Length; arrayIndex++)
            {
  
                if (convertedActual[arrayIndex] == classIndex && convertedPredicted[arrayIndex] == classIndex)
                {
                    tp += 1;
                }  else if (convertedActual[arrayIndex] == classIndex && convertedPredicted[arrayIndex] != classIndex)
                {
                    fn += 1;
                } else if (convertedActual[arrayIndex] != classIndex && convertedPredicted[arrayIndex] == classIndex)
                {
                    fp += 1;
                }
                else
                {
                    fp += 1;
                }

            }
            statisticalMetrics.AssignConfusionMatrixCoefficients(classIndex, tp, tn, fp, fn);
            
        }

        return statisticalMetrics;
    }

}
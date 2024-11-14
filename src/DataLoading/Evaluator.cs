namespace DataLoading;

public class Evaluator(float maxError = 0.0001f)
{

    // Checks if one prediction was correct
    public bool EvaluateCompleteResult(float[] predicted, float[] actual)
    {
        return predicted.
            Select((value, index) => MathF.Abs(value - actual[index]) <= maxError)
            .All(condition => condition);
    }

    // Converts vectors like (0.0, 0.1, 0.2, 0.1, 0.98, 0.12, 0.11, 0.11, 0.3, 0.4) to number (4)
    public int[] ConvertToClasses(float[][] toConvert)
    {
        return toConvert.Select(value => Array.IndexOf(value, value.Max())).ToArray();
    }

    // Returns vector, where true if correct prediction, false otherwise
    public bool[] EvaluateCompletedResults(float[][] predicted, float[][] actual)
    {
        return predicted.Select((value, index) => EvaluateCompleteResult(value, actual[index])).ToArray();
    }

    // From all predicted and all actual vectors computes evaluation scores
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
                    tp++;
                }  else if (convertedActual[arrayIndex] == classIndex && convertedPredicted[arrayIndex] != classIndex)
                {
                    fn++;
                } else if (convertedActual[arrayIndex] != classIndex && convertedPredicted[arrayIndex] == classIndex)
                {
                    fp++;
                }
                else
                {
                    tn++;
                }

            }
            statisticalMetrics.FillMetrics(classIndex, tp, tn, fp, fn);
            
        }

        return statisticalMetrics;
    }

}
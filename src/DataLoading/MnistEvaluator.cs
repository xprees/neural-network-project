namespace DataLoading;

public class MnistEvaluator(float maxError = 0.0001f)
{
    /// Checks if one prediction was correct
    public bool EvaluateCompleteResult(float[] predicted, int actualIndex) =>
        Array.IndexOf(predicted, predicted.Max()) == actualIndex;

    /// Converts vectors like (0.0, 0.1, 0.2, 0.1, 0.98, 0.12, 0.11, 0.11, 0.3, 0.4) to number (4)
    public int[] ConvertToClasses(float[][] toConvert) =>
        toConvert.Select(value => Array.IndexOf(value, value.Max())).ToArray();

    /// Returns vector, where true if correct prediction, false otherwise
    public bool[] EvaluateCompletedResults(float[][] predicted, float[] actual) =>
        predicted
            .Select((value, index) => EvaluateCompleteResult(value, (int)actual[index]))
            .ToArray();

    /// From all predicted and all actual vectors computes evaluation scores
    public StatisticalMetrics EvaluateModel(float[][] predicted, float[] actual)
    {
        var statisticalMetrics = new StatisticalMetrics();
        var convertedPredicted = ConvertToClasses(predicted);
        var convertedActual = Array.ConvertAll(actual, x => (int)x);
        for (var classIndex = 0; classIndex < 10; classIndex++) // Not the most effective, can be remade later
        {
            var fp = 0;
            var fn = 0;
            var tp = 0;
            var tn = 0;

            for (var arrayIndex = 0; arrayIndex < convertedPredicted.Length; arrayIndex++)
            {
                if (convertedActual[arrayIndex] == classIndex && convertedPredicted[arrayIndex] == classIndex)
                {
                    tp++;
                    continue;
                }

                if (convertedActual[arrayIndex] == classIndex && convertedPredicted[arrayIndex] != classIndex)
                {
                    fn++;
                    continue;
                }

                if (convertedActual[arrayIndex] != classIndex && convertedPredicted[arrayIndex] == classIndex)
                {
                    fp++;
                    continue;
                }

                tn++;
            }

            statisticalMetrics.FillMetric(classIndex, tp, tn, fp, fn);
        }

        statisticalMetrics.ComputeMetrics();
        return statisticalMetrics;
    }
}
namespace DataProcessing.Evaluation;

public class AccuracyEvaluator(float classificationTolerance = 0.01f) : IEvaluator<float>
{
    public float Evaluate(float[][] predicted, float[][] expected)
    {
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("The number of predicted and expected values must be the same");
        }

        var positive = 0;
        var total = 0;
        for (var i = 0; i < predicted.Length; i++)
        {
            if (predicted[i].Length != expected[i].Length)
            {
                throw new ArgumentException("The number of predicted and expected values must be the same");
            }

            for (var j = 0; j < predicted[i].Length; j++)
            {
                if (Math.Abs(predicted[i][j] - expected[i][j]) < classificationTolerance)
                {
                    positive++;
                }

                total++;
            }
        }

        return (float)positive / total;
    }
}
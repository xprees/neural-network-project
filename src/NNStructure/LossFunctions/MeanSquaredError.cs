namespace NNStructure.LossFunctions;

public class MeanSquaredError : ILossFunction
{
    public float Calculate(float[] predicted, float[] expected)
    {
        if (predicted.Length != expected.Length)
            throw new ArgumentException("The length of predicted and expected arrays must be the same.");

        var sum = 0f;
        for (var i = 0; i < predicted.Length; i++)
        {
            var diff = predicted[i] - expected[i];
            sum += diff * diff;
        }

        return sum / predicted.Length;
    }

    public float[] CalculateGradient(float[] predicted, float[] expected)
    {
        throw new NotImplementedException();
    }
}
namespace NNStructure.LossFunctions;

public class CrossEntropy : ILossFunction
{
    private const float Epsilon = 1e-8f; // To avoid log(0)

    public float Calculate(float[] predicted, float[] expected)
    {
        float loss = 0;
        for (var i = 0; i < predicted.Length; i++)
        {
            loss -= expected[i] * MathF.Log(predicted[i] + Epsilon);
        }

        return loss;
    }

    public float[] CalculateGradient(float[] predicted, float[] expected)
    {
        // Softmax activation function is used in the output layer (predicted values were passed through it)
        var gradient = new float[predicted.Length];
        for (var i = 0; i < predicted.Length; i++)
        {
            gradient[i] = predicted[i] - expected[i];
        }

        return gradient;
    }
}
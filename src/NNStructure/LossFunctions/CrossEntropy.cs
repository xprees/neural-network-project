using NNStructure.ActivationFunctions;

namespace NNStructure.LossFunctions;

public class CrossEntropy : ILossFunction
{
    private const float Epsilon = 0.00000001f; // To avoid log(0)
    private readonly Softmax _softmax = new();

    public float ComputeCrossEntropyOnVector(float[] predictedVector, int actual) =>
        -MathF.Log(predictedVector[actual] + Epsilon);

    /// Applies SoftMax on predicted vector and computes CrossEntropy
    public float CrossEntropyVector(float[] predictedVector, int actual) =>
        ComputeCrossEntropyOnVector(_softmax.ActivateLayer(predictedVector), actual);


    public float Calculate(float[] predicted, float[] expected)
    {
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("The length of predicted and expected arrays must be the same.");
        }

        if (predicted.Length <= 0) return 0;

        var max = expected.Max();
        var index = expected.ToList().IndexOf(max);
        return CrossEntropyVector(predicted, index);
    }


    public float[] CalculateGradient(float[] predicted, float[] expected)
    {
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("The length of predicted and expected arrays must be the same.");
        }

        // Softmax activation function is used in the output layer (predicted values were passed through it)
        var gradient = new float[predicted.Length];
        for (var i = 0; i < predicted.Length; i++)
        {
            gradient[i] = predicted[i] - expected[i];
        }

        return gradient;
    }
}
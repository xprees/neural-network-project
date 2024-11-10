namespace NNStructure.LossFunctions;

/// Implements Mean Squared Error (MSE) as presented in slides <a href="https://is.muni.cz/auth/el/fi/podzim2024/PV021/um/NEW_lecture_slides__continuously_updated.pdf">PV021 - slide 232</a>
/// <remarks>Implemented with <b>1/2 * MSE</b> for better differentiation properties</remarks>
public class MeanSquaredError : ILossFunction
{
    public float Calculate(float[] predicted, float[] expected)
    {
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("The length of predicted and expected arrays must be the same.");
        }

        if (predicted.Length <= 0) return 0;

        var squaredSum = 0f;
        for (var i = 0; i < predicted.Length; i++)
        {
            var diff = predicted[i] - expected[i];
            squaredSum += diff * diff;
        }

        return 0.5f * squaredSum / predicted.Length; // 0.5 for simpler gradient calculation
    }

    public float[] CalculateGradient(float[] predicted, float[] expected)
    {
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("The length of predicted and expected arrays must be the same.");
        }

        var gradients = new float[predicted.Length];
        for (var i = 0; i < predicted.Length; i++)
        {
            gradients[i] = (predicted[i] - expected[i]) / predicted.Length; // Gradient of 0.5 * MSE
        }

        return gradients;
    }
}
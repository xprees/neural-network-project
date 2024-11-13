namespace NNStructure.ActivationFunctions;

/// Hyperbolic Tangent activation function.
public class Tanh : IActivationFunction
{
    public float Activate(float potential) => (float)Math.Tanh(potential);

    public float Derivative(float value)
    {
        var tanhX = (float)Math.Tanh(value);
        return 1 - tanhX * tanhX;
    }
}
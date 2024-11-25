namespace NNStructure.ActivationFunctions;

/// Hyperbolic Tangent activation function.
public class Tanh : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials) =>
        potentials
            .Select(potential => (float)Math.Tanh(potential))
            .ToArray();

    public float Derivative(float innerPottential)
    {
        var tanhX = (float)Math.Tanh(innerPottential);
        return 1 - tanhX * tanhX;
    }
}
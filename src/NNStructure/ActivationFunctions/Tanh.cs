namespace NNStructure.ActivationFunctions;

/// Hyperbolic Tangent activation function.
public class Tanh : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials) =>
        potentials
            .Select(potential => (float)Math.Tanh(potential))
            .ToArray();

    public float[] DerivativePotentials(float[] innerPotentials) =>
        innerPotentials
            .Select(TanhX)
            .ToArray();

    private static float TanhX(float innerPotential)
    {
        var tanhX = (float)Math.Tanh(innerPotential);
        return 1 - tanhX * tanhX;
    }
}
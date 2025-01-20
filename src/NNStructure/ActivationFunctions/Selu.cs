namespace NNStructure.ActivationFunctions;

// https://paperswithcode.com/method/selu
public class Selu : IActivationFunction
{
    private const float Alpha = 1.6733f;
    private const float Scale = 1.0507f;

    public float[] ActivateLayer(float[] potentials)
    {
        return potentials
            .Select(CalculateActivation)
            .ToArray();
    }

    private float CalculateActivation(float potential)
    {
        if (potential >= 0)
        {
            return Scale * potential;
        }

        return Scale * Alpha * (MathF.Exp(potential) - 1);
    }

    public float[] DerivativePotentials(float[] innerPotentials)
    {
        return innerPotentials
            .Select(p => p >= 0 ? Scale : Scale * Alpha * MathF.Exp(p))
            .ToArray();
    }
}
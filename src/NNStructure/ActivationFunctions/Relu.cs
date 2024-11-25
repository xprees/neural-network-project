namespace NNStructure.ActivationFunctions;

public class Relu : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials) =>
        potentials
            .Select(potential => Math.Max(0, potential))
            .ToArray();

    public float[] DerivativePotentials(float[] innerPotentials) =>
        innerPotentials
            .Select(p => p > 0 ? 1f : 0f)
            .ToArray();
}
namespace NNStructure.ActivationFunctions;

public class Relu : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials) =>
        potentials
            .Select(potential => Math.Max(0, potential))
            .ToArray();

    public float Derivative(float innerPottential) =>
        innerPottential > 0 ? 1 : 0;
}
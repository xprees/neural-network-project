namespace NNStructure.ActivationFunctions;

public class Relu : IActivationFunction
{
    public float Activate(float potential) =>
        Math.Max(0, potential);

    public float Derivative(float value) =>
        value > 0 ? 1 : 0;
}
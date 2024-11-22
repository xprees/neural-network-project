namespace NNStructure.ActivationFunctions;

public class Sigmoid : IActivationFunction
{
    public float Activate(float potential) =>
        1 / (1 + (float)Math.Exp(-potential));

    public float Derivative(float value) =>
        Activate(value) * (1 - Activate(value));
}
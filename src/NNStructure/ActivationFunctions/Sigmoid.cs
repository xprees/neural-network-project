namespace NNStructure.ActivationFunctions;

public class Sigmoid : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials) =>
        potentials
            .Select(ActivateSingleNeuron)
            .ToArray();

    private float ActivateSingleNeuron(float potential) =>
        1 / (1 + (float)Math.Exp(-potential));

    public float Derivative(float value) =>
        ActivateSingleNeuron(value) * (1 - ActivateSingleNeuron(value));
}
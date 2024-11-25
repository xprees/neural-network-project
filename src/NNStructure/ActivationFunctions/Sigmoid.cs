namespace NNStructure.ActivationFunctions;

public class Sigmoid : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials) =>
        potentials
            .Select(ActivateSingleNeuron)
            .ToArray();

    private float ActivateSingleNeuron(float potential) =>
        1 / (1 + (float)Math.Exp(-potential));

    public float[] DerivativePotentials(float[] innerPotentials) =>
        innerPotentials
            .Select(DerivativePotential)
            .ToArray();

    private float DerivativePotential(float innerPotential)
    {
        var activation = ActivateSingleNeuron(innerPotential);
        return activation * (1 - activation);
    }
}
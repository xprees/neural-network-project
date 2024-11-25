namespace NNStructure.ActivationFunctions;

public class Softmax : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials)
    {
        var max = potentials.Max();
        var softMax = potentials
            //.Select(x => x - max) // Subtracting the maximum value to prevent overflow
            .Select(MathF.Exp)
            .ToArray();
        var sum = softMax.Sum();
        return softMax.Select(x => x / sum).ToArray();
    }


    public float Derivative(float innerPottential)
    {
        return 1; // TODO implement
    }
}
namespace NNStructure.ActivationFunctions;

public class Softmax : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials)
    {
        var max = potentials.Max();
        var softMax = potentials
            .Select(x => x - max) // Subtracting the maximum value to prevent overflow
            .Select(MathF.Exp)
            .ToArray();
        var sum = softMax.Sum();
        return softMax.Select(x => x / sum).ToArray();
    }


    public float[] DerivativePotentials(float[] innerPotentials)
    {
        var softmax = ActivateLayer(innerPotentials);
        var derivative = new float[innerPotentials.Length];

        for (var i = 0; i < innerPotentials.Length; i++)
        {
            derivative[i] = softmax[i] * (1 - softmax[i]);
        }

        return derivative;
    }
}
namespace NNStructure.ActivationFunctions;

public class Softmax : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials)
    {
        var max = potentials.Max();
        var expPotentials = new float[potentials.Length];
        var sum = 0f;

        for (var i = 0; i < potentials.Length; i++)
        {
            // Subtracting the maximum value to prevent overflow
            expPotentials[i] = MathF.Exp(potentials[i] - max);
            sum += expPotentials[i];
        }

        for (var i = 0; i < expPotentials.Length; i++)
        {
            expPotentials[i] /= sum;
        }

        return expPotentials;
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
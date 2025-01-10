namespace NNStructure.ActivationFunctions;

public class LeakyRelu : IActivationFunction
{
    public float[] ActivateLayer(float[] potentials)
    {
        var activated = new float[potentials.Length];
        for (var i = 0; i < potentials.Length; i++)
        {
            potentials[i] = Math.Max(0.01f * potentials[i], potentials[i]);
        }

        return activated;
    }

    public float[] DerivativePotentials(float[] innerPotentials)
    {
        var derivatives = new float[innerPotentials.Length];
        for (var i = 0; i < innerPotentials.Length; i++)
        {
            derivatives[i] = innerPotentials[i] > 0 ? 1 : 0.01f;
        }

        return derivatives;
    }
}
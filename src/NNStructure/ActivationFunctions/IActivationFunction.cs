namespace NNStructure.ActivationFunctions;

public interface IActivationFunction
{
    /// Returns the activation of the layer.
    float[] ActivateLayer(float[] potentials);

    float[] DerivativePotentials(float[] innerPotentials);
}
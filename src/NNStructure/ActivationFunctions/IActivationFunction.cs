namespace NNStructure.ActivationFunctions;

public interface IActivationFunction
{
    float Activate(Neuron neuron);
    float Derivative(float value);
}
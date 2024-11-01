namespace NNStructure.ActivationFunctions;

public interface IActivationFunction
{
    float Activate(float potential);
    float Derivative(float value);
}
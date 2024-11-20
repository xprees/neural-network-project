namespace NNStructure.LossFunctions;

public interface ILossFunction
{
    float Calculate(float[] predicted, float[] expected);
    float[] CalculateGradient(float[] predicted, float[] expected);
}
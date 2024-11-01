namespace NNStructure.LossFunctions;

public interface ILossFunction
{
    float Calculate(float[] predicted, float[] actual);
    float[] CalculateGradient(float[] predicted, float[] actual);
}
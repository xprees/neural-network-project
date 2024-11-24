using System.ComponentModel.DataAnnotations;
using NNStructure.LossFunctions;

namespace NNStructure.CrossEntropy;

public class CrossEntropy: ILossFunction
{
    
    private float ComputeCrossEntropyOnVector(float[] predictedVector, int actual)
    {
        return - MathF.Log(predictedVector[actual]);
    }
    
    
    private float[] ComputeSoftMaxOnVector(float[] predictedVector)
    {
        float[] softMax = predictedVector.Select(x => (float)Math.Exp(x)).ToArray();
        float sum = softMax.Sum();
        return softMax.Select(x => x/sum).ToArray();
    }
    
    
    // Applies SoftMax on predicted vector and computes CrossEntropy
    private float CrossEntropyVector(float[] predictedVector, int actual)
    {
        return ComputeCrossEntropyOnVector(ComputeSoftMaxOnVector(predictedVector), actual);
    }
    
    

    public float Calculate(float[] predicted, float[] expected)
    {
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("The length of predicted and expected arrays must be the same.");
        }

        if (predicted.Length <= 0) return 0;
        
        int index = expected.ToList().IndexOf(expected.Max());
        return CrossEntropyVector(predicted, index);
    }
    

    public float[] CalculateGradient(float[] predicted, float[] expected)
    {
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("The length of predicted and expected arrays must be the same.");
        }

        var gradients = ComputeSoftMaxOnVector(predicted);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] -= expected[i];
        }

        return gradients;
    }
}
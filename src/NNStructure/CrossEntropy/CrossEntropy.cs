namespace NNStructure.CrossEntropy;

public class CrossEntropy
{
    public static float ComputeCrossEntropyOnVector(float[] predictedVector, int actual)
    {
        return - MathF.Log(predictedVector[actual]);
    }

    public static float ComputeCrossEntropyOnBatch(float[][] predictedVectors, int[] actual, int batchSize)
    {
        float result = 0f;
        for (int index = 0; index < batchSize; index++)
        {
            result += ComputeCrossEntropyOnVector(predictedVectors[index], actual[index]);
        }
        return result / batchSize;
    }

    public static float[] ComputeSoftMaxOnVector(float[] predictedVector)
    {
        float[] softMax = predictedVector.Select(x => (float)Math.Exp(x)).ToArray();
        float sum = softMax.Sum();
        return softMax.Select(x => x/sum).ToArray();
    }
    
    public static float[][] ComputeSoftMaxOnBatch(float[][] predictedVectors)
    {
        return predictedVectors.Select(ComputeSoftMaxOnVector).ToArray();
    }
}
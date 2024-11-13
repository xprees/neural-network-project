namespace DataLoading;

public class Evaluator(string path, float maxError = 0.0001f)
{
    private readonly DataLoader _resultLoader = new(path);

    public bool EvaluateNext(float predictedResult)
    {
        var vector = _resultLoader.ReadOneVector();
        if (vector == null)
        {
            throw new NullReferenceException("Read result is null");
        }
        return MathF.Abs(predictedResult - vector[0]) < maxError;

    }

    public bool?[] EvaluateBatch(float?[] predictedResults, int batchSize)
    {
        float[]?[] vectors = _resultLoader.ReadNVectors(batchSize);
        

        return predictedResults
            .Select((value, i) =>
                value == null || vectors[i] == null 
                    ? null
                    : (bool?)(MathF.Abs((float)(value - vectors[i][0])) < maxError))
            .ToArray();
        
    }

    public bool?[] EvaluateAll(float?[] predictedResults)
    {
        float[]?[] vectors = _resultLoader.ReadAllVectors();
        return predictedResults
            .Select((value, i) =>
                value == null || vectors[i] == null 
                    ? null
                    : (bool?)(MathF.Abs((float)(value - vectors[i][0])) < maxError))
            .ToArray();
    }

    public static float GetMax(float?[] values)
    {
        float? max = values.Where(x => x.HasValue).Max();
        if (!max.HasValue)
        {
            throw new NullReferenceException("Array was empty!");
        }
        else
        {
            return max.Value;
        }
        
    }
}
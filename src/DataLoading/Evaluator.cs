namespace DataLoading;

public class Evaluator(string path, float maxError = 0.0001f)
{
    private readonly DataLoader _resultLoader = new(path);

    /// <summary>
    /// Evaluates given float result with next in line from file specified in constructor
    /// </summary>
    /// <param name="predictedResult">float with predicted value</param>
    /// <returns>true if predictedResult and loaded label differ less than specified error</returns>
    /// <exception cref="NullReferenceException">no value to read</exception>
    public bool EvaluateNext(float predictedResult)
    {
        var vector = _resultLoader.ReadOneVector();
        if (vector == null)
        {
            throw new NullReferenceException("Read result is null");
        }
        return MathF.Abs(predictedResult - vector[0]) < maxError;

    }

    /// <summary>
    /// Evaluates given predicted results with n lines from file specified in constructor
    /// </summary>
    /// <param name="predictedResults">float?[] of predicted numbers</param>
    /// <param name="batchSize">--</param>
    /// <returns>bool array -> true if values differ less than maximal error</returns>
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
/// <summary>
/// Evaluates predictedResults with whole file of labels
/// </summary>
/// <param name="predictedResults">array of all predicted Values</param>
/// <returns>bool array -> true if values differ less than maximal error</returns>
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

/// <summary>
/// Helper function for getting maximal value from array
/// </summary>
/// <param name="values">array</param>
/// <returns>maximal value</returns>
/// <exception cref="NullReferenceException">if there is no value in the array</exception>
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
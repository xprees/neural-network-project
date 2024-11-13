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
}
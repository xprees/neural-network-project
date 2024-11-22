namespace DataProcessing.Evaluation;

public interface IEvaluator<out T>
{
    /// Evaluate the predicted values against the expected values
    public T Evaluate(float[][] predicted, float[][] expected);
}
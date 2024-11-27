namespace NNStructure.Optimizers;

/// ADAM - Adaptive Moment Estimation optimizer. <a href="https://paperswithcode.com/method/adam">Papers with code</a>
/// <remarks>Learning rate is recommend around 1e-3 and Beta1 and Beta2 forgetting params typically around 0.9 and 0.999</remarks>
public class Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f) : IOptimizer
{
    private int _timeStep;

    public float LearningRate { get; set; } = learningRate;

    public float Beta1 { get; set; } = beta1;

    public float Beta2 { get; set; } = beta2;

    public float UpdateWeight(float weight, float gradient, ref float velocity, ref float squareGradient)
    {
        _timeStep++;

        velocity = Beta1 * velocity + (1 - Beta1) * gradient; // m(t)
        var mHat = velocity / (1 - MathF.Pow(Beta1, _timeStep));

        squareGradient = Beta2 * squareGradient + (1 - Beta2) * gradient * gradient; // v(t)
        var vHat = squareGradient / (1 - MathF.Pow(Beta2, _timeStep));

        return weight - LearningRate * mHat / (MathF.Sqrt(vHat) + 1e-8f);
    }
}
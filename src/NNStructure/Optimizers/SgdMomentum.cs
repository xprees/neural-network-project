namespace NNStructure.Optimizers;

/// Stochastic Gradient Descent with momentum optimizer. <a href="https://paperswithcode.com/method/sgd-with-momentum">Papers with code</a>
/// <remarks>generally use the value of β like 0.9,0.99 or 0.5 only</remarks>
public class SgdMomentum(float learningRate, float momentum) : IOptimizer
{
    public float LearningRate { get; set; } = learningRate;
    public float Momentum { get; set; } = momentum;

    public float UpdateWeight(float weight, float gradient, ref float velocity)
    {
        velocity = Momentum * velocity + LearningRate * gradient;
        return weight - velocity;
    }
}
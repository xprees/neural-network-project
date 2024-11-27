namespace NNStructure.Optimizers;

/// Root Mean Square Propagation optimizer. <a href="https://paperswithcode.com/method/rmsprop">Papers with code</a>
/// <remarks>Hinton suggest decayRate 0.9 and LearningRate 0.001</remarks>
public class RmsProp(float learningRate, float decayRate) : IOptimizer
{
    public float LearningRate { get; set; } = learningRate;

    public float DecayRate { get; set; } = decayRate;

    public float UpdateWeight(float weight, float gradient, ref float velocity, ref float squareGradient)
    {
        squareGradient = DecayRate * squareGradient + (1 - DecayRate) * gradient * gradient;
        return weight - LearningRate * gradient / MathF.Sqrt(squareGradient + 1e-8f);
    }
}
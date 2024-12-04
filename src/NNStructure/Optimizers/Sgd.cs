namespace NNStructure.Optimizers;

/// Stochastic Gradient Descent optimizer with constant learning rate.
public class Sgd(float learningRate) : IOptimizer
{
    public float LearningRate { get; set; } = learningRate;

    public int TimeStep { get; set; } // Not used in SGD

    public float UpdateWeight(float weight, float gradient, ref float velocity, ref float squaredGradient) =>
        weight - LearningRate * gradient;
}
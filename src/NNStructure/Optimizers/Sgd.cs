namespace NNStructure.Optimizers;

/// Stochastic Gradient Descent optimizer with constant learning rate.
public class Sgd(float learningRate) : IOptimizer
{
    public float LearningRate { get; set; } = learningRate;

    public float UpdateWeight(float weight, float gradient, ref float extraData) =>
        weight - LearningRate * gradient;
}
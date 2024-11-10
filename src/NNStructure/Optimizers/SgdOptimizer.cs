namespace NNStructure.Optimizers;

/// Stochastic Gradient Descent optimizer with constant learning rate.
public class SgdOptimizer(float learningRate) : IOptimizer
{
    public float LearningRate { get; set; } = learningRate;

    public float UpdateWeight(float weight, float gradient, int batchSize) =>
        weight - LearningRate * gradient;
}
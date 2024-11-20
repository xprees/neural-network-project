namespace NNStructure.Optimizers;

/// Interface for all optimizers used in the Neural Network, e.g. Stochastic Gradient Descent, Adam, etc.
/// They keep their state between the epochs and update the weights of the network such as  learning rate, momentum, etc.
/// <remarks>If optimizer has inner state that changes make sure that accessing/updating it is thread-safe</remarks>
public interface IOptimizer
{
    public float LearningRate { get; set; }

    /// Updates the weight of the network using the gradient and batch size. Remember to divide the gradient by the batch size.
    float UpdateWeight(float weight, float gradient);
}
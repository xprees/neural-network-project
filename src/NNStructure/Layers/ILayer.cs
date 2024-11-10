using System.Collections.Concurrent;
using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Optimizers;

namespace NNStructure.Layers;

public interface ILayer
{
    int InputSize { get; }

    int OutputSize { get; }

    float[] Biases { get; set; }

    Neuron[] Neurons { get; set; }

    float[][] Weights { get; set; }

    IActivationFunction ActivationFunction { get; set; }

    /// Initializes all weights in the Layer using initializer
    void InitializeWeights(IWeightsInitializer initializer);

    /// Resets the gradients of Neurons in the layer
    void ResetGradients();

    void AggregateGradients(ConcurrentDictionary<int, float[][]> gradientsByTrainingExample, int batchSize);

    /// Apply all neuron gradients to the weights of the layer
    void UpdateWeights(IOptimizer optimizer);

    float[] DoForwardPass(float[] input);

    float[] DoBackpropagation(float[] values, ref float[] batchGradients);
}
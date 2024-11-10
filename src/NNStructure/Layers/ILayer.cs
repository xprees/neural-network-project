using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Optimizers;

namespace NNStructure.Layers;

public interface ILayer
{
    int InputSize { get; }

    int OutputSize { get; }

    Neuron[] Neurons { get; set; }

    /// Weights of the layer. First dimension is the number of neurons in the layer, second dimension is the number of inputs to the layer + 1 w0 is Bias.
    float[,] Weights { get; set; }

    IActivationFunction ActivationFunction { get; }

    /// Initializes all weights in the Layer using initializer
    void InitializeWeights(IWeightsInitializer initializer);

    /// Resets the gradients of Neurons in the layer
    void ResetGradients();

    /// Apply all neuron gradients to the weights of the layer
    void UpdateWeights(float[] layerGradients, IOptimizer optimizer, int batchSize);

    float[] DoForwardPass(float[] input);

    float[] DoBackpropagation(float[] topLayerGradient, ref float[] batchGradients);
}
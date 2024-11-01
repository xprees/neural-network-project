using NNStructure.ActivationFunctions;
using NNStructure.Initialization;

namespace NNStructure.Layers;

public interface ILayer
{
    int InputSize { get; }

    int OutputSize { get; }

    float[] Biases { get; set; }

    Neuron[] Neurons { get; set; }

    float[][] Weights { get; set; }

    IActivationFunction ActivationFunction { get; set; }

    void InitializeWeights(IWeightsInitializer initializer);

    void UpdateWeights(float learningRate);

    float[] DoForwardPass(float[] input);

    float[] DoBackpropagation(float[] values);
}
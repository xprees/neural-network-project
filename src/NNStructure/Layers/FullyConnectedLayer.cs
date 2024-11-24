using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Optimizers;

namespace NNStructure.Layers;

public class FullyConnectedLayer(int inputSize, int outputSize, IActivationFunction activationFn) : ILayer
{
    public int InputSize { get; } = inputSize;
    public int OutputSize { get; } = outputSize;
    public float[,] Weights { get; set; } = new float[outputSize, inputSize + 1]; // +1 for bias on index 0
    public IActivationFunction ActivationFunction { get; } = activationFn;

    public void InitializeWeights(IWeightsInitializer initializer)
    {
        for (var i = 0; i < OutputSize; i++)
        {
            for (var j = 0; j < InputSize + 1; j++)
            {
                Weights[i, j] = initializer.GetInitialWeight(this);
            }
        }
    }

    public void UpdateWeights(float[,] layerGradients, IOptimizer optimizer, int batchSize)
    {
        for (var i = 0; i < OutputSize; i++)
        {
            for (var j = 0; j < InputSize + 1; j++) // including bias on index 0
            {
                Weights[i, j] = optimizer.UpdateWeight(Weights[i, j], layerGradients[i, j]);
            }
        }
    }

    public (float[] output, float[] innerPotentials) DoForwardPass(float[] input)
    {
        var innerPotentials = new float[OutputSize]; // Inner potentials of neurons
        Parallel.For(0, OutputSize, i =>
            {
                var innerPotential = Weights[i, 0]; // Bias
                for (var j = 0; j < InputSize; j++)
                {
                    innerPotential += Weights[i, j + 1] * input[j]; // +1 to skip bias
                }

                innerPotentials[i] = innerPotential;
            }
        );

        var output = ActivationFunction.ActivateLayer(innerPotentials);
        return (output, innerPotentials);
    }

    public float[] DoBackpropagation(float[] topLayerGradient, float[] layerInput,
        float[] layerInnerPotentials, ref float[,] layerBatchGradients)
    {
        // Initialize gradients array for this layer and this training example
        layerBatchGradients = new float[OutputSize, InputSize + 1];

        var inputGradients = new float[InputSize];

        for (var i = 0; i < OutputSize; i++)
        {
            var activationDerivative = ActivationFunction.Derivative(layerInnerPotentials[i]);
            var gradient = topLayerGradient[i] * activationDerivative;

            layerBatchGradients[i, 0] = gradient; // Bias
            for (var j = 1; j < InputSize + 1; j++) // Start from 1 to skip bias
            {
                layerBatchGradients[i, j] = gradient * layerInput[j - 1];

                inputGradients[j - 1] += Weights[i, j] * gradient;
            }
        }

        return inputGradients;
    }
}
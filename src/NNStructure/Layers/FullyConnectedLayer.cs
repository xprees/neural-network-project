using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Optimizers;

namespace NNStructure.Layers;

public class FullyConnectedLayer(
    int inputSize,
    int outputSize,
    IActivationFunction activationFn,
    float overrideLearningRate = 0) : ILayer
{
    private readonly int _inputSize = inputSize;
    private readonly int _outputSize = outputSize;

    public int InputSize { get; } = inputSize;
    public int OutputSize { get; } = outputSize;

    public float[,,] Weights { get; set; } = new float[outputSize, inputSize + 1, 3];
    // +1 for bias on index 0; (0 - weight, 1 - velocity/square gradient - for Momentum, 2 - square gradient - for Adam/RMSProp)

    public IActivationFunction ActivationFunction { get; } = activationFn;

    public void InitializeWeights(IWeightsInitializer initializer)
    {
        for (var i = 0; i < _outputSize; i++)
        {
            for (var j = 0; j < _inputSize + 1; j++)
            {
                Weights[i, j, 0] = initializer.GetInitialWeight(this);
            }
        }
    }

    public void ResetStateBeforeNewBatchRun()
    {
        for (var i = 0; i < _outputSize; i++)
        {
            for (var j = 0; j < _inputSize + 1; j++)
            {
                Weights[i, j, 1] = 0; // Reset velocity
                Weights[i, j, 2] = 0; // Reset square gradient
            }
        }
    }

    public void UpdateWeights(float[,] layerGradients, IOptimizer optimizer, int batchSize)
    {
        var previousLearningRate = optimizer.LearningRate;
        if (overrideLearningRate > 0) optimizer.LearningRate = overrideLearningRate;

        for (var i = 0; i < _outputSize; i++)
        {
            for (var j = 0; j < _inputSize + 1; j++) // including bias on index 0
            {
                Weights[i, j, 0] = optimizer.UpdateWeight(Weights[i, j, 0], layerGradients[i, j],
                    ref Weights[i, j, 1], ref Weights[i, j, 2]);
            }
        }

        optimizer.LearningRate = previousLearningRate;
    }

    public (float[] output, float[] potentialGradients) DoForwardPass(float[] input)
    {
        var innerPotentials = new float[_outputSize]; // Inner potentials of neurons
        Parallel.For(0, _outputSize, i =>
            {
                var innerPotential = Weights[i, 0, 0]; // Bias
                for (var j = 0; j < _inputSize; j++)
                {
                    innerPotential += Weights[i, j + 1, 0] * input[j]; // +1 to skip bias
                }

                innerPotentials[i] = innerPotential;
            }
        );

        var output = ActivationFunction.ActivateLayer(innerPotentials);
        var potentialGradients = ActivationFunction.DerivativePotentials(innerPotentials);
        return (output, potentialGradients);
    }

    public float[] DoBackpropagation(float[] topLayerGradient, float[] layerInput,
        float[] potentialGradients, ref float[,] layerBatchGradients)
    {
        // Initialize gradients array for this layer and this training example
        layerBatchGradients = new float[_outputSize, _inputSize + 1];
        var inputGradients = new float[_inputSize];

        for (var i = 0; i < _outputSize; i++)
        {
            var activationDerivative = potentialGradients[i];
            var gradient = topLayerGradient[i] * activationDerivative;

            layerBatchGradients[i, 0] = gradient; // Bias
            for (var j = 1; j < _inputSize + 1; j++) // Start from 1 to skip bias
            {
                layerBatchGradients[i, j] = gradient * layerInput[j - 1];

                inputGradients[j - 1] += Weights[i, j, 0] * gradient;
            }
        }

        return inputGradients;
    }
}
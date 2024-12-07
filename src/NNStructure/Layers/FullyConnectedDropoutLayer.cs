using NNStructure.ActivationFunctions;

namespace NNStructure.Layers;

public class FullyConnectedDropoutLayer(
    int inputSize,
    int outputSize,
    IActivationFunction activationFn,
    float dropoutRate = 0.5f,
    float overrideLearningRate = 0,
    int seed = 42) : FullyConnectedLayer(inputSize, outputSize, activationFn, overrideLearningRate)
{
    private readonly Random _random = new(seed);
    private readonly int _inputSize = inputSize;
    private readonly int _outputSize = outputSize;
    private readonly bool[] _dropoutMask = new bool[outputSize];

    public new void ResetStateBeforeNewBatchRun()
    {
        for (var i = 0; i < _outputSize; i++)
        {
            _dropoutMask[i] = _random.NextSingle() > dropoutRate; // Generate mask
            for (var j = 0; j < _inputSize + 1; j++)
            {
                Weights[i, j, 1] = 0; // Reset velocity
                Weights[i, j, 2] = 0; // Reset square gradient
            }
        }
    }

    public new (float[] output, float[] potentialGradients) DoForwardPass(float[] input, bool isTraining)
    {
        var innerPotentials = new float[_outputSize]; // Inner potentials of neurons
        Parallel.For(0, _outputSize, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
            {
                // Neuron will be dropped - no need to calculate inner potential
                if (isTraining && !_dropoutMask[i]) return;

                var innerPotential = Weights[i, 0, 0]; // Bias
                for (var j = 0; j < _inputSize; j++)
                {
                    innerPotential += Weights[i, j + 1, 0] * input[j]; // +1 to skip bias
                }

                // Apply Inverted dropout scaling
                if (isTraining)
                {
                    innerPotential /= (1 - dropoutRate);
                }

                innerPotentials[i] = innerPotential;
            }
        );

        var output = ActivationFunction.ActivateLayer(innerPotentials);
        // Gradients of inner potentials -> precomputed for backpropagation
        var innerPotentialGradients = ActivationFunction.DerivativePotentials(innerPotentials);

        if (!isTraining) return (output, innerPotentialGradients);

        // Apply dropout mask on output during training
        for (var i = 0; i < _outputSize; i++)
        {
            if (!_dropoutMask[i])
            {
                output[i] = 0;
            }
        }

        return (output, innerPotentialGradients);
    }

    public new float[] DoBackpropagation(float[] topLayerGradient, float[] layerInput,
        float[] innerPotentialGradients, ref float[,] layerBatchGradients)
    {
        // Initialize gradients array for this layer and this training example
        layerBatchGradients = new float[_outputSize, _inputSize + 1];
        var inputGradients = new float[_inputSize];

        for (var i = 0; i < _outputSize; i++)
        {
            if (!_dropoutMask[i]) continue; // Neuron is dropped

            var topGradient = topLayerGradient[i] / (1 - dropoutRate);
            var gradient = topGradient * innerPotentialGradients[i];

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
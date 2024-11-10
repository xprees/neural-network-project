using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Optimizers;

namespace NNStructure.Layers;

public class FullyConnectedLayer(int inputSize, int outputSize, IActivationFunction activationFn) : ILayer
{
    public int InputSize { get; } = inputSize;
    public int OutputSize { get; } = outputSize;
    public Neuron[] Neurons { get; set; } = new Neuron[outputSize];
    public float[,] Weights { get; set; } = new float[outputSize, inputSize + 1]; // +1 for bias on index 0
    public IActivationFunction ActivationFunction { get; } = activationFn;

    public void InitializeWeights(IWeightsInitializer initializer)
    {
        for (var i = 0; i < OutputSize; i++)
        {
            Neurons[i] = new Neuron();
            for (var j = 0; j < InputSize + 1; j++)
            {
                Weights[i, j] = initializer.GetInitialWeight(this);
            }
        }
    }

    public void ResetGradients()
    {
        for (var i = 0; i < Neurons.Length; i++)
        {
            var neuron = Neurons[i];
            Neurons[i] = neuron with { Gradient = 0 };
        }
    }

    public void UpdateWeights(float[] layerGradients, IOptimizer optimizer, int batchSize)
    {
        for (var i = 0; i < OutputSize; i++)
        {
            for (var j = 0; j < InputSize + 1; j++) // including bias on index 0
            {
                Weights[i, j] = optimizer.UpdateWeight(Weights[i, j], layerGradients[i], batchSize);
            }
        }
    }

    public float[] DoForwardPass(float[] input)
    {
        var output = new float[OutputSize];
        for (var i = 0; i < OutputSize; i++)
        {
            var neuron = Neurons[i];
            var innerPotential = Weights[i, 0]; // Bias
            for (var j = 0; j < InputSize; j++)
            {
                innerPotential += Weights[i, j + 1] * input[j]; // +1 to skip bias
            }

            var activationValue = ActivationFunction.Activate(innerPotential);
            output[i] = activationValue;

            // TODO this won't work in parallel (changing shared state) -> use a temporary array or concurrent dictionary
            // Is it really necessary to have the neurons?
            Neurons[i] = neuron with { InnerPotential = innerPotential, ActivationValue = activationValue };
        }

        return output;
    }

    public float[] DoBackpropagation(float[] topLayerGradient, ref float[] batchGradients)
    {
        var inputGradients = new float[InputSize];
        batchGradients = new float[OutputSize];

        for (var i = 0; i < OutputSize; i++)
        {
            var neuron = Neurons[i];
            var activationDerivative = ActivationFunction.Derivative(neuron.InnerPotential);
            var delta = topLayerGradient[i] * activationDerivative;

            batchGradients[i] = delta;

            for (var j = 1; j < InputSize + 1; j++) // Start from 1 to skip bias
            {
                inputGradients[j - 1] += Weights[i, j] * delta;
            }
        }

        return inputGradients;
    }
}
using System.Collections.Concurrent;
using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

namespace NNStructure;

public class NeuralNetwork(ILossFunction lossFunction, IWeightsInitializer initializer, IOptimizer optimizer)
{
    public List<ILayer> Layers { get; } = [];

    public void AddLayer(ILayer layer) => Layers.Add(layer);

    public void InitializeWeights()
    {
        foreach (var layer in Layers)
        {
            layer.InitializeWeights(initializer);
        }
    }

    /// Does the forward propagation for the input vector and returns prediction vector 
    public float[] ForwardPropagate(float[] input)
    {
        CheckInputDimensionMatchesFirstLayerOrThrow(input);

        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.DoForwardPass(output);
        }

        return output;
    }

    public float[][] BackPropagate(float[] predictedOutput, float[] expectedOutput)
    {
        var lossGradient = lossFunction.CalculateGradient(predictedOutput, expectedOutput);
        var batchGradients = new float[Layers.Count][];

        for (var i = Layers.Count - 1; i >= 0; i--)
        {
            var layer = Layers[i];
            lossGradient = layer.DoBackpropagation(lossGradient, ref batchGradients[i]);
        }

        return batchGradients;
    }


    public void Train(float[][] inputs, float[][] expectedResults, int maxEpochs, int miniBatchSize)
    {
        for (var epoch = 0; epoch < maxEpochs; epoch++)
        {
            Layers.ForEach(l => l.ResetGradients());

            var miniBatch = ChooseMiniBatch(inputs, expectedResults, epoch, miniBatchSize)
                .ToArray();
            var gradientsByTrainingExample =
                new ConcurrentDictionary<int, float[][]>(); // Gradients for each training example by k their index

            for (var k = 0; k < miniBatch.Length; k++)
            {
                var (trainingExample, expectedResult) = miniBatch[k];
                var predictedOutput = ForwardPropagate(trainingExample);

                var kthBatchGradients = BackPropagate(predictedOutput, expectedResult);
                if (!gradientsByTrainingExample.TryAdd(k, kthBatchGradients))
                {
                    throw new InvalidOperationException("Failed to add gradients to the dictionary.");
                }
            }

            var layersGradients = AggregateGradientsByLayers(gradientsByTrainingExample);
            for (var i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var layerGradients = layersGradients[i];
                layer.UpdateWeights(layerGradients, optimizer, miniBatchSize);
            }
        }
    }

    private float[][] AggregateGradientsByLayers(ConcurrentDictionary<int, float[][]> gradientsByTrainingExample)
    {
        if (gradientsByTrainingExample.IsEmpty)
        {
            throw new InvalidOperationException("No gradients calculated for updating.");
        }

        var layersGradients = new float[Layers.Count][];
        for (var i = 0; i < Layers.Count; i++)
        {
            layersGradients[i] = new float[gradientsByTrainingExample.Values.FirstOrDefault()?[i].Length ?? 0];
        }

        foreach (var kthExampleGradients in gradientsByTrainingExample.Values)
        {
            for (var layerIndex = 0; layerIndex < Layers.Count; layerIndex++)
            {
                for (var i = 0; i < kthExampleGradients[layerIndex].Length; i++)
                {
                    layersGradients[layerIndex][i] += kthExampleGradients[layerIndex][i];
                }
            }
        }

        if (layersGradients.Length != Layers.Count)
        {
            throw new InvalidOperationException(
                $"The number of layers ({Layers.Count}) and the number of gradients ({layersGradients.Length}) do not match."
            );
        }


        return layersGradients;
    }

    /// Chooses a mini batch of examples from the training set.
    /// <remarks>Cycles through dataset if it runs out of new cases</remarks>
    private IEnumerable<(float[] trainingExample, float[] expectedResults)> ChooseMiniBatch(float[][] inputs,
        float[][] expectedResults, int epoch,
        int miniBatchSize)
    {
        var inputsLength = inputs.Length;
        var startIndex = epoch * miniBatchSize;

        var examplesCount = 0;
        while (examplesCount <= miniBatchSize)
        {
            yield return (inputs[startIndex % inputsLength], expectedResults[startIndex % inputsLength]);
            startIndex++;
            examplesCount++;
        }
    }

    private void CheckInputDimensionMatchesFirstLayerOrThrow(float[] input)
    {
        var firstLayerInputLength = Layers.FirstOrDefault()?.InputSize;
        if (firstLayerInputLength != input.Length)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) has to match first layer input size ({firstLayerInputLength})!"
            );
        }
    }
}
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
    public (float[] prediction, float[][] layerInputs) ForwardPropagate(float[] input)
    {
        CheckInputDimensionMatchesFirstLayerOrThrow(input);

        var layerInputs = new float[Layers.Count][];
        var output = input;
        for (var i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];

            layerInputs[i] = output;
            output = layer.DoForwardPass(output);
        }

        return (output, layerInputs);
    }

    public float[][,] BackPropagate(float[] predictedOutput, float[] expectedOutput, float[][] layerInputs)
    {
        var lossGradient = lossFunction.CalculateGradient(predictedOutput, expectedOutput);
        var batchGradients = new float[Layers.Count][,];

        for (var i = Layers.Count - 1; i >= 0; i--)
        {
            var layer = Layers[i];
            lossGradient = layer.DoBackpropagation(lossGradient, layerInputs[i], ref batchGradients[i]);
        }

        return batchGradients;
    }


    public void Train(float[][] inputs, float[][] expectedResults, int maxEpochs, int miniBatchSize)
    {
        for (var epoch = 0; epoch < maxEpochs; epoch++)
        {
            Layers.ForEach(l => l.ResetStateBeforeEpochRun());

            var miniBatch = ChooseMiniBatch(inputs, expectedResults, epoch, miniBatchSize)
                .ToArray();
            var gradientsByTrainingExample =
                new ConcurrentDictionary<int, float[][,]>(); // Gradients for each training example by k their index

            for (var k = 0; k < miniBatch.Length; k++)
            {
                var (trainingExample, expectedResult) = miniBatch[k];
                var (predictedOutput, layerInputs) = ForwardPropagate(trainingExample);

                var kthBatchGradients = BackPropagate(predictedOutput, expectedResult, layerInputs);
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

    private float[][,] AggregateGradientsByLayers(ConcurrentDictionary<int, float[][,]> gradientsByTrainingExample)
    {
        if (gradientsByTrainingExample.IsEmpty)
        {
            throw new InvalidOperationException("No gradients calculated for updating.");
        }

        var layersGradients = new float[Layers.Count][,];
        for (var i = 0; i < Layers.Count; i++)
        {
            var firstExampleGradients = gradientsByTrainingExample.Values.FirstOrDefault();
            if (firstExampleGradients == null)
            {
                throw new InvalidOperationException("No gradients found for the first example.");
            }

            layersGradients[i] =
                new float[firstExampleGradients[i].GetLength(0), firstExampleGradients[i].GetLength(1)];
        }

        foreach (var kthExampleGradients in gradientsByTrainingExample.Values)
        {
            for (var layerIndex = 0; layerIndex < Layers.Count; layerIndex++)
            {
                for (var i = 0; i < kthExampleGradients[layerIndex].GetLength(0); i++)
                {
                    for (var j = 0; j < kthExampleGradients[layerIndex].GetLength(1); j++)
                    {
                        // Average the gradients for each layer 
                        layersGradients[layerIndex][i, j] +=
                            kthExampleGradients[layerIndex][i, j] / gradientsByTrainingExample.Count;
                    }
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
        while (examplesCount < miniBatchSize)
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
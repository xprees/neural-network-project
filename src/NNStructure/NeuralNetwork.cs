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


    public void Train(float[][] inputs, float[][] expectedResults, int epochs, int miniBatchSize)
    {
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            Layers.ForEach(l => l.ResetGradients());

            var miniBatch = ChooseMiniBatch(inputs, epoch, miniBatchSize);
            var gradientsByTrainingExample =
                new ConcurrentDictionary<int, float[][]>(); // Gradients for each training example by k their index

            for (var k = 0; k < inputs.Length; k++)
            {
                var trainingExample = miniBatch[k];
                var predictedOutput = ForwardPropagate(trainingExample);

                var kthBatchGradients = BackPropagate(predictedOutput, expectedResults[k]);
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
        var layersGradients = new float[Layers.Count][];
        foreach (var kthExampleGradients in gradientsByTrainingExample.Values)
        {
            // TODO implement aggregation -> need to transpose the data
        }

        if (layersGradients.Length != Layers.Count)
        {
            throw new InvalidOperationException(
                $"The number of layers ({Layers.Count}) and the number of gradients ({layersGradients.Length}) do not match."
            );
        }

        return layersGradients;
    }

    private float[][] ChooseMiniBatch(float[][] inputs, int epoch, int miniBatchSize) =>
        inputs
            .Skip(epoch * miniBatchSize)
            .Take(miniBatchSize)
            .ToArray();
}
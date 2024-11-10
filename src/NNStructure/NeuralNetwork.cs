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

            foreach (var layer in Layers)
            {
                layer.AggregateGradients(gradientsByTrainingExample, miniBatchSize);
                layer.UpdateWeights(optimizer);
            }
        }
    }

    private float[][] ChooseMiniBatch(float[][] inputs, int epoch, int miniBatchSize) =>
        inputs
            .Skip(epoch * miniBatchSize)
            .Take(miniBatchSize)
            .ToArray();
}
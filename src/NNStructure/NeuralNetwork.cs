using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

namespace NNStructure;

public record EpochEndEventArgs(int Epoch);

public class NeuralNetwork(
    ILossFunction lossFunction,
    IWeightsInitializer initializer,
    IOptimizer optimizer,
    int seed = 42)
{
    private readonly Random _random = new(seed);
    public List<ILayer> Layers { get; } = [];

    public event EventHandler<EpochEndEventArgs>? OnEpochEnd;

    public void AddLayer(ILayer layer) => Layers.Add(layer);

    public void InitializeWeights()
    {
        foreach (var layer in Layers)
        {
            layer.InitializeWeights(initializer);
        }
    }

    /// Does the forward propagation for the input vector and returns prediction vector 
    public (float[] prediction, float[][] layerInputs, float[][] layersInnerPotentials) ForwardPropagate(float[] input)
    {
        var layerInputs = new float[Layers.Count][];
        var layersInnerPotentials = new float[Layers.Count][];
        var output = input;
        for (var i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];

            layerInputs[i] = output;
            var (layerOutput, innerPotentials) = layer.DoForwardPass(output);
            output = layerOutput;
            layersInnerPotentials[i] = innerPotentials;
        }

        return (output, layerInputs, layersInnerPotentials);
    }

    public float[][,] BackPropagate(float[] predictedOutput,
        float[] expectedOutput, float[][] layerInputs, float[][] potentialGradients)
    {
        var lossGradient = lossFunction.CalculateGradient(predictedOutput, expectedOutput);
        var batchGradients = new float[Layers.Count][,];

        for (var i = Layers.Count - 1; i >= 0; i--)
        {
            var layer = Layers[i];
            lossGradient = layer.DoBackpropagation(lossGradient, layerInputs[i], potentialGradients[i],
                ref batchGradients[i]);
        }

        return batchGradients;
    }

    public void Train(float[][] inputs, float[][] expectedResults, int maxEpochs, int miniBatchSize,
        bool preEpochDataShuffle = true)
    {
        var miniBatchRuns = inputs.Length / miniBatchSize;
        for (var epoch = 0; epoch < maxEpochs; epoch++)
        {
            if (preEpochDataShuffle)
            {
                (inputs, expectedResults) = ShuffleData(inputs, expectedResults);
            }

            for (var miniBatchRun = 0; miniBatchRun < miniBatchRuns; miniBatchRun++)
            {
                Layers.ForEach(l => l.ResetStateBeforeNewBatchRun());
                var run = miniBatchRun + miniBatchRuns * epoch;
                var miniBatch = ChooseMiniBatch(inputs, expectedResults, run, miniBatchSize)
                    .ToArray();

                // Gradients for each training example by k their index
                var gradientsByTrainingExample = new float[miniBatchSize][][,];

                Parallel.For(0, miniBatch.Length, new ParallelOptions { MaxDegreeOfParallelism = 16 }, k =>
                {
                    var (trainingExample, expectedResult) = miniBatch[k];
                    var (predictedOutput, layerInputs, potentialGradients) = ForwardPropagate(trainingExample);

                    var kthBatchGradients =
                        BackPropagate(predictedOutput, expectedResult, layerInputs, potentialGradients);

                    gradientsByTrainingExample[k] = kthBatchGradients;
                });

                var layersGradients = AggregateGradientsByLayers(gradientsByTrainingExample, miniBatchSize);
                for (var i = 0; i < Layers.Count; i++)
                {
                    var layer = Layers[i];
                    var layerGradients = layersGradients[i];
                    layer.UpdateWeights(layerGradients, optimizer, miniBatchSize);
                }

                optimizer.TimeStep++;
            }

            OnEpochEnd?.Invoke(this, new EpochEndEventArgs(epoch));
        }
    }

    private (float[][] inputs, float[][] expectedResults) ShuffleData(float[][] inputs, float[][] expectedResults)
    {
        var zippedData = inputs
            .Zip(expectedResults, (input, expected) => (input, expected))
            .OrderBy(_ => _random.Next())
            .ToArray();
        inputs = zippedData.Select(z => z.input).ToArray();
        expectedResults = zippedData.Select(z => z.expected).ToArray();
        return (inputs, expectedResults);
    }

    /// Does the forward propagation for the input vectors and returns prediction vectors
    public float[][] Test(float[][] inputs)
    {
        var predictions = new float[inputs.Length][];
        for (var i = 0; i < inputs.Length; i++)
        {
            predictions[i] = ForwardPropagate(inputs[i]).prediction;
        }

        return predictions;
    }

    private float[][,] AggregateGradientsByLayers(float[][][,] gradientsByTrainingExample, int miniBatchSize)
    {
        var firstExampleGradients = gradientsByTrainingExample[0];

        // Initialize the array with zeros
        var layersCount = Layers.Count;
        var layersGradients = new float[layersCount][,];
        for (var layerIndex = 0; layerIndex < layersCount; layerIndex++)
        {
            var neuronsCount = firstExampleGradients[layerIndex].GetLength(0);
            var weightsCount = firstExampleGradients[layerIndex].GetLength(1);
            layersGradients[layerIndex] = new float[neuronsCount, weightsCount];
        }


        foreach (var kthExampleGradients in gradientsByTrainingExample)
        {
            for (var layerIndex = 0; layerIndex < layersCount; layerIndex++)
            {
                var neuronsCount = layersGradients[layerIndex].GetLength(0);
                for (var i = 0; i < neuronsCount; i++)
                {
                    var weightsCount = layersGradients[layerIndex].GetLength(1);
                    for (var j = 0; j < weightsCount; j++)
                    {
                        // Average the gradients for each layer 
                        layersGradients[layerIndex][i, j] +=
                            kthExampleGradients[layerIndex][i, j] / (float)miniBatchSize;
                    }
                }
            }
        }

        return layersGradients;
    }

    /// Chooses a mini batch of examples from the training set.
    /// <remarks>Cycles through dataset if it runs out of new cases</remarks>
    private IEnumerable<(float[] trainingExample, float[] expectedResults)> ChooseMiniBatch(
        float[][] inputs, float[][] expectedResults, int miniBatchRun, int miniBatchSize
    )
    {
        var inputsLength = inputs.Length;
        var startIndex = miniBatchRun * miniBatchSize;

        var examplesCount = 0;
        while (examplesCount < miniBatchSize)
        {
            yield return (inputs[startIndex % inputsLength], expectedResults[startIndex % inputsLength]);
            startIndex++;
            examplesCount++;
        }
    }
}
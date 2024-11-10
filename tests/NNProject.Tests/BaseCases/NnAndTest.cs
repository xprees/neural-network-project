using FluentAssertions;
using NNStructure;
using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

namespace NNProjectTests.BaseCases;

[TestFixture]
public class NnAndTest
{
    private NeuralNetwork _nn;

    #region Data

    private static readonly float[][] TestInputs =
    [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ];

    private static readonly float[][] ExpectedResults =
    [
        [1],
        [0],
        [0],
        [0]
    ];

    #endregion

    [SetUp]
    public void Setup()
    {
        _nn = new NeuralNetwork(new MeanSquaredError(), new RandomWeightInitializer(), new SgdOptimizer(0.005f));
        _nn.AddLayer(new FullyConnectedLayer(2, 2, new Relu()));
        _nn.AddLayer(new FullyConnectedLayer(2, 1, new Relu()));
    }

    [TestCase(1, 5)]
    [TestCase(2, 5)]
    [TestCase(3, 5)]
    [TestCase(5, 5)]
    public void TestNnAndTraining(int miniBatchSize, int maxEpochs)
    {
        _nn.InitializeWeights();

        _nn.Train(TestInputs, ExpectedResults, maxEpochs, miniBatchSize);

        const float precision = 0.25f;

        // 1, 1 = 1
        var result = _nn.ForwardPropagate([1, 1]).First();
        result.Should().BeApproximately(1, precision, $"For [1, 1] expected 1, got {result}");

        // 0, 0 = 0
        result = _nn.ForwardPropagate([0, 0]).First();
        result.Should().BeApproximately(0, precision, $"For [0, 0] expected 0, got {result}");

        // 1, 0 = 0
        result = _nn.ForwardPropagate([1, 0]).First();
        result.Should().BeApproximately(0, precision, $"For [1, 0] expected 0, got {result}");

        // 0, 1 = 0
        result = _nn.ForwardPropagate([0, 1]).First();
        result.Should().BeApproximately(0, precision, $"For [0, 1] expected 0, got {result}");
    }
}
using FluentAssertions;
using NNStructure;
using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

namespace NNProjectTests.BaseCases;

[TestFixture]
public class NnXorTests
{
    private NeuralNetwork _nn;

    #region Data

    private static readonly float[][] _testInputs =
    [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0]
    ];

    private static readonly float[][] _expectedResults =
    [
        [0],
        [1],
        [1],
        [0]
    ];

    #endregion

    [SetUp]
    public void Setup()
    {
        _nn = new NeuralNetwork(new MeanSquaredError(), new RandomValueInitializer(), new SgdOptimizer(0.3f));
        _nn.AddLayer(new FullyConnectedLayer(2, 2, new Relu()));
        _nn.AddLayer(new FullyConnectedLayer(2, 1, new Relu()));
    }

    [Test]
    [Explicit]
    public void TestNnXorTraining()
    {
        _nn.InitializeWeights();

        _nn.Train(_testInputs, _expectedResults, 4, 2);

        // 1, 1 = 0
        var result = _nn.ForwardPropagate([1, 1]).First(); // Single output neuron
        result.Should().BeApproximately(0, 0.1f, $"For [1, 1] expected 0, got {result}");

        // 0, 0 = 0
        result = _nn.ForwardPropagate([0, 0]).First(); // Single output neuron
        result.Should().BeApproximately(0, 0.1f, $"For [0, 0] expected 0, got {result}");

        // 1, 0 = 1
        result = _nn.ForwardPropagate([1, 0]).First(); // Single output neuron
        result.Should().BeApproximately(1, 0.1f, $"For [1, 0] expected 1, got {result}");

        // 0, 1 = 1
        result = _nn.ForwardPropagate([0, 1]).First(); // Single output neuron
        result.Should().BeApproximately(1, 0.1f, $"For [0, 1] expected 1, got {result}");
    }
}
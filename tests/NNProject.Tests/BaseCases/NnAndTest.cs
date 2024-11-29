using DataProcessing.Evaluation;
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
    private const float ClassificationTolerance = 0.15f;
    private const float ExpectedAccuracy = 0.999f;

    private NeuralNetwork _nn;
    private AccuracyEvaluator _accuracyEvaluator;

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
        _accuracyEvaluator = new AccuracyEvaluator(ClassificationTolerance);
        _nn = new NeuralNetwork(new MeanSquaredError(), new GlorotWeightInitializer(), new Adam(0.001f));
        _nn.AddLayer(new FullyConnectedLayer(2, 2, new Tanh()));
        _nn.AddLayer(new FullyConnectedLayer(2, 1, new Tanh()));
    }

    [TestCase(4, 1000)]
    public void TestNnAndTraining(int miniBatchSize, int maxEpochs)
    {
        _nn.InitializeWeights();

        _nn.Train(TestInputs, ExpectedResults, maxEpochs, miniBatchSize);

        var predicted = _nn.Test(TestInputs);
        var accuracy = _accuracyEvaluator.Evaluate(predicted, ExpectedResults);

        accuracy.Should().BeGreaterThan(ExpectedAccuracy,
            $"The accuracy of the model should be greater than {ExpectedAccuracy} but was {accuracy}");
    }
}
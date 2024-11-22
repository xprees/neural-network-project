using DataProcessing.Evaluation;
using FluentAssertions;

namespace DataLoadingTests.Evaluation;

[TestFixture]
public class AccuracyEvaluatorTests
{
    private AccuracyEvaluator _evaluator;

    [SetUp]
    public void Setup()
    {
        _evaluator = new AccuracyEvaluator();
    }

    [Test]
    public void EvaluateAccuracy_WithCorrectPredictions_Returns1()
    {
        float[][] predictions = [[1f], [1f], [1f]];
        float[][] expected = [[1f], [1f], [1f]];

        var accuracy = _evaluator.Evaluate(predictions, expected);

        accuracy.Should().Be(1);
    }

    [Test]
    public void EvaluateAccuracy_WithIncorrectPredictions_Returns0()
    {
        float[][] predictions = [[0f], [0f], [0f]];
        float[][] expected = [[1f], [1f], [1f]];

        var accuracy = _evaluator.Evaluate(predictions, expected);

        accuracy.Should().Be(0);
    }

    [Test]
    public void EvaluateAccuracy_WithMixedPredictions_ReturnsCorrectResult()
    {
        float[][] predictions = [[1f], [0f], [1f]];
        float[][] expected = [[1f], [1f], [1f]];

        var accuracy = _evaluator.Evaluate(predictions, expected);

        accuracy.Should().BeApproximately(0.6667f, 0.01f);
    }
}
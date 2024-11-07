using FluentAssertions;
using NNStructure.LossFunctions;

namespace NNStructureTests.LossFunctions;

[TestFixture]
public class MeanSquaredErrorTests
{
    private MeanSquaredError _squaredError;

    [SetUp]
    public void Setup()
    {
        _squaredError = new MeanSquaredError();
    }

    [Test]
    public void CalculateSimpleZeroError()
    {
        float[] expected = [1f];
        float[] predicted = [1f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().Be(0);
    }

    [Test]
    public void CalculateSimpleNonZeroError()
    {
        float[] expected = [1f];
        float[] predicted = [0f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().Be(1);
    }

    [Test]
    public void CalculateMultipleValuesError()
    {
        float[] expected = [1f, 2f, 3f];
        float[] predicted = [1f, 2f, 2f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().BeApproximately(0.333f, 0.001f);
    }

    [Test]
    public void CalculateZeroLengthArrays()
    {
        float[] expected = [];
        float[] predicted = [];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().Be(0);
    }

    [Test]
    public void CalculateNegativeValuesError()
    {
        float[] expected = [-1f, -2f, -3f];
        float[] predicted = [-1f, -2f, -2f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().BeApproximately(0.333f, 0.001f);
    }

    [Test]
    public void CalculateMixedValuesError()
    {
        float[] expected = [1f, -2f, 3f];
        float[] predicted = [1f, -2f, 2f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().BeApproximately(0.333f, 0.001f);
    }

    [Test]
    public void CalculateLargeValuesError()
    {
        float[] expected = [1000f, 2000f, 3000f];
        float[] predicted = [1000f, 2000f, 2000f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().BeApproximately(333333.333f, 0.001f);
    }
}
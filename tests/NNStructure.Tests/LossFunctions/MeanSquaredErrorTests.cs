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

    #region Error

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

        error.Should().Be(0.5f);
    }

    [Test]
    public void CalculateMultipleValuesError()
    {
        float[] expected = [1f, 2f, 3f];
        float[] predicted = [1f, 2f, 2f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().BeApproximately(0.333f / 2f, 0.001f);
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

        error.Should().BeApproximately(0.333f / 2f, 0.001f);
    }

    [Test]
    public void CalculateMixedValuesError()
    {
        float[] expected = [1f, -2f, 3f];
        float[] predicted = [1f, -2f, 2f];

        var error = _squaredError.Calculate(predicted, expected);

        error.Should().BeApproximately(0.333f / 2f, 0.001f);
    }

    #endregion

    #region Gradient

    [Test]
    public void CalculateGradientSimpleZeroError()
    {
        float[] expected = [1f];
        float[] predicted = [1f];

        var gradient = _squaredError.CalculateGradient(predicted, expected);

        gradient.Should().Equal(0f / 1);
    }

    [Test]
    public void CalculateGradientSimpleNonZeroError()
    {
        float[] expected = [1f];
        float[] predicted = [0f];

        var gradient = _squaredError.CalculateGradient(predicted, expected);

        gradient.Should().Equal(-1f / 1);
    }

    [Test]
    public void CalculateGradientMultipleValuesError()
    {
        float[] expected = [1f, 2f, 3f];
        float[] predicted = [1f, 2f, 2f];

        var gradient = _squaredError.CalculateGradient(predicted, expected);

        gradient.Should().Equal(0f, 0f, -1f / 3f);
    }

    [Test]
    public void CalculateGradientNegativeValuesError()
    {
        float[] expected = [-1f, -2f, -3f];
        float[] predicted = [-1f, -2f, -2f];

        var gradient = _squaredError.CalculateGradient(predicted, expected);

        gradient.Should().Equal(0f, 0f, 1f / 3f); // Divide by length
    }

    [Test]
    public void CalculateGradientMixedValuesError()
    {
        float[] expected = [1f, -2f, 3f];
        float[] predicted = [1f, -2f, 2f];

        var gradient = _squaredError.CalculateGradient(predicted, expected);

        gradient.Should().Equal(0f, 0f, -1f / 3f);
    }

    [Test]
    public void CalculateGradientLargeValuesError()
    {
        float[] expected = [1000f, 2000f, 3000f];
        float[] predicted = [1000f, 2000f, 2000f];

        var gradient = _squaredError.CalculateGradient(predicted, expected);

        gradient.Should().Equal(0f, 0f, -1000f / 3f);
    }

    #endregion
}
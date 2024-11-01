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
}
using FluentAssertions;
using NNStructure.LossFunctions;

namespace NNStructureTests.LossFunctions;

[TestFixture]
public class CrossEntropyTests
{
    [Test]
    public void CalculateGradientTest()
    {
        float[] predicted = [0.1f, 0.2f, 0.7f];
        float[] expected = [0f, 0f, 1f];
        var entropy = new CrossEntropy();
        var gradient = entropy.CalculateGradient(predicted, expected);

        float[] expectedGradient = [0.1f, 0.2f, -0.3f];
        gradient.Should().BeEquivalentTo(expectedGradient, options => options.WithStrictOrdering());
    }
}
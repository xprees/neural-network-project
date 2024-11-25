using FluentAssertions;
using NNStructure.LossFunctions;

namespace NNStructureTests.LossFunctions;

[TestFixture]
public class CrossEntropyTests
{
    [Test]
    public void CrossEntropyTest()
    {
        float[] predicted = [-2f, -1f, -0.1f, 10f, -10f, -0f, 0f, 0.000001f, -0.000001f, -0.000001f];
        var entropy = new CrossEntropy();

        var entropies = new float[10];
        var minEntropy = 1000f;

        for (var i = 0; i < predicted.Length; i++)
        {
            var crossEntropy = entropy.CrossEntropyVector(predicted, i);
            Console.WriteLine($"{i}. {crossEntropy} was predicted... ");
            entropies[i] = crossEntropy;
            if (crossEntropy < minEntropy)
            {
                minEntropy = crossEntropy;
            }
        }

        entropies[3].Should().Be(minEntropy);
    }

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
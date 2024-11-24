using FluentAssertions;
using NNStructure.CrossEntropy;

namespace NNStructureTests.LossFunctions;

public class CrossEntropyTests
{
    [Test]
    public void SoftMaxTestSum()
    {
        float[] predicted = [-2f ,-1f , -0.1f, 100f, -100f, -0f, 0f, 0.000001f, -0.000001f, -0.000001f];
        CrossEntropy entropy = new CrossEntropy();
        predicted = entropy.ComputeSoftMaxOnVector(predicted);
        predicted.Sum().Should().BeApproximately(1.0f, 0.000001f);
        Console.WriteLine(predicted.Sum() + " was predicted... ");

    }
}
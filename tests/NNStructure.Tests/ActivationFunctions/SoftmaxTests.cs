using FluentAssertions;
using NNStructure.ActivationFunctions;

namespace NNStructureTests.ActivationFunctions;

[TestFixture]
public class SoftmaxTests
{
    [Test]
    public void SoftMaxTestSum()
    {
        float[] predicted = [-2f, -1f, -0.1f, 10f, -10f, -0f, 0f, 0.000001f, -0.000001f, -0.000001f];

        predicted = new Softmax().ActivateLayer(predicted);

        predicted.Sum().Should().BeApproximately(1.0f, 0.000001f);
    }
}
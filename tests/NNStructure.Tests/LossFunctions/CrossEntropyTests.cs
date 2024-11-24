using FluentAssertions;
using NNStructure.CrossEntropy;

namespace NNStructureTests.LossFunctions;

public class CrossEntropyTests
{
    [Test]
    public void SoftMaxTestSum()
    {
        float[] predicted = [-2f ,-1f , -0.1f, 10f, -10f, -0f, 0f, 0.000001f, -0.000001f, -0.000001f];
        CrossEntropy entropy = new CrossEntropy();
        predicted = entropy.ComputeSoftMaxOnVector(predicted);
        predicted.Sum().Should().BeApproximately(1.0f, 0.000001f);
        Console.WriteLine(predicted.Sum() + " was predicted... ");
        for (int i = 0; i < predicted.Length; i++)
        {
            Console.WriteLine(predicted[i] + " was " + predicted[i]);
        }

    }
    
    [Test]
    public void CrossEntropyTest()
    {
        float[] predicted = [-2f ,-1f , -0.1f, 10f, -10f, -0f, 0f, 0.000001f, -0.000001f, -0.000001f];
        CrossEntropy entropy = new CrossEntropy();
        float[] entropies = new float[10];
        float minEntropy = 1000f;
        for (int i = 0; i < predicted.Length; i++)
        {
            float crossEntropy = entropy.CrossEntropyVector(predicted, i);
            Console.WriteLine(i + ". " + crossEntropy + " was predicted... ");
            entropies[i] = crossEntropy;
            if (crossEntropy < minEntropy)
            {
                minEntropy = crossEntropy;
            }
        }

        entropies[3].Should().Be(minEntropy);


    }
}
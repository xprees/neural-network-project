using NNStructure.Layers;

namespace NNStructure.Initialization;

/// Glorot (Xavier) weight initializer for initializing weights in the neural network.
public class GlorotWeightInitializer(int seed = 42) : IWeightsInitializer
{
    private readonly Random _random = new(seed);

    public float GetInitialWeight(ILayer initializedLayer)
    {
        var inputSize = initializedLayer.InputSize;
        var outputSize = initializedLayer.OutputSize;
        var limit = MathF.Sqrt(6 / (float)(inputSize + outputSize));
        return (_random.NextSingle() * 2 - 1) * limit;
    }
}
using NNStructure.Layers;

namespace NNStructure.Initialization;

public class RandomValueInitializer(int seed = 42) : IWeightsInitializer
{
    private readonly Random _random = new(seed);

    public float GetInitialWeight(ILayer initializedLayer) =>
        _random.NextSingle() - 0.5f;
}
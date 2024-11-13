using NNStructure.Layers;

namespace NNStructure.Initialization;

public class RandomWeightInitializer(int seed = 42) : IWeightsInitializer
{
    private readonly Random _random = new(seed);

    public float GetInitialWeight(ILayer initializedLayer) =>
        (_random.NextSingle() - 0.5f) * 0.1f;
}
using NNStructure.Layers;

namespace NNStructure.Initialization;

public class RandomValueInitializer() : IWeightsInitializer
{
    private readonly Random _random = Random.Shared;

    public float GetInitialWeight(ILayer initializedLayer) =>
        _random.NextSingle() - 0.5f;
}
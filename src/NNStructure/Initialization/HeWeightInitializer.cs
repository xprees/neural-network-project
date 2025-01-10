using NNStructure.Layers;

namespace NNStructure.Initialization;

public class HeWeightInitializer(int seed = 42) : IWeightsInitializer
{
    private readonly Random _random = new(seed);

    public float GetInitialWeight(ILayer initializedLayer)
    {
        return RandomInNormalDistribution.GetRandomInNormalDistribution(_random,
            1.0f / MathF.Sqrt(initializedLayer.InputSize));
    }
}
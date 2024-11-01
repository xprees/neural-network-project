using NNStructure.Layers;

namespace NNStructure.Initialization;

public interface IWeightsInitializer
{
    float GetInitialWeight(ILayer initializedLayer);
}
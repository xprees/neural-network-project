using BenchmarkDotNet.Attributes;

namespace NN.Benchmarking.ActivationFunctions;

[SimpleJob]
[MemoryDiagnoser]
public class SoftmaxActivationBenchmark
{
    private readonly Random _random = new(42);

    private float[] _testPotentials = null!;

    [Params(128, 512, 784)] public int InputSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _testPotentials = new float[InputSize];
        for (var i = 0; i < InputSize; i++)
        {
            _testPotentials[i] = _random.NextSingle();
        }
    }

    [Benchmark(Baseline = true)]
    public float[] ActivateLayerLinq()
    {
        var max = _testPotentials.Max();
        var softMax = _testPotentials
            .Select(x => x - max) // Subtracting the maximum value to prevent overflow
            .Select(MathF.Exp)
            .ToArray();
        var sum = softMax.Sum();
        return softMax.Select(x => x / sum).ToArray();
    }

    [Benchmark]
    public float[] ActivateLayerFor()
    {
        var max = _testPotentials.Max();
        var expPotentials = new float[_testPotentials.Length];
        var sum = 0f;

        for (var i = 0; i < _testPotentials.Length; i++)
        {
            expPotentials[i] = MathF.Exp(_testPotentials[i] - max);
            sum += expPotentials[i];
        }

        for (var i = 0; i < expPotentials.Length; i++)
        {
            expPotentials[i] /= sum;
        }

        return expPotentials;
    }
}
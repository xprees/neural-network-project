using BenchmarkDotNet.Attributes;
using NNStructure.ActivationFunctions;

namespace NN.Benchmarking.ActivationFunctions;

[SimpleJob]
[MemoryDiagnoser]
public class ReluActivationBenchmark

{
    private readonly Random _random = new(42);
    private readonly Softmax _softmax = new();

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
    public float[] ActivateLinq() =>
        _testPotentials
            .Select(potential => Math.Max(0, potential))
            .ToArray();

    [Benchmark]
    public float[] ActivateFor()
    {
        var result = new float[_testPotentials.Length];
        for (var i = 0; i < _testPotentials.Length; i++)
        {
            result[i] = Math.Max(0, _testPotentials[i]);
        }

        return result;
    }
}
using BenchmarkDotNet.Attributes;
using NNStructure.ActivationFunctions;

namespace NN.Benchmarking.ActivationFunctions;

[SimpleJob]
public class SoftmaxDeriveBenchmark
{
    private readonly Random _random = new(42);
    private readonly Softmax _softmax = new();

    private float[] _testPotentials = null!;
    private ParallelOptions _parallelOptions = null!;

    [Params(128, 512, 784)] public int InputSize { get; set; }

    [Params(16, 32)] public int Parallelism { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _testPotentials = new float[InputSize];
        for (var i = 0; i < InputSize; i++)
        {
            _testPotentials[i] = _random.NextSingle();
        }

        _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Parallelism };
    }

    [Benchmark(Baseline = true)]
    public float[] DerivativePotentialsSync()
    {
        var softmax = _softmax.ActivateLayer(_testPotentials);
        var derivative = new float[_testPotentials.Length];

        for (var i = 0; i < _testPotentials.Length; i++)
        {
            derivative[i] = softmax[i] * (1 - softmax[i]);
        }

        return derivative;
    }

    [Benchmark]
    public float[] DerivativePotentialsParallel()
    {
        var softmax = _softmax.ActivateLayer(_testPotentials);
        var derivative = new float[_testPotentials.Length];

        Parallel.For(0, _testPotentials.Length, _parallelOptions, i => derivative[i] = softmax[i] * (1 - softmax[i]));

        return derivative;
    }
}
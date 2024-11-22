using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace NN.Benchmarking.LossFunctions;

[SimpleJob(RuntimeMoniker.Net80)]
[SimpleJob(RuntimeMoniker.NativeAot80)]
public class MeanSquaredErrorBenchmarks
{
    // 28 * 28 -> 784
    [Params(256, 512, 784, 1024, 4096)] public int InputSize { get; set; }

    private float[] _predictedValues = null!;
    private float[] _expectedValues = null!;

    [GlobalSetup]
    public void GlobalSetup()
    {
        _predictedValues = new float[InputSize];
        _expectedValues = new float[InputSize];
    }

    [Benchmark]
    public float CalculateMseSync()
    {
        var squaredSum = 0f;
        for (var i = 0; i < _predictedValues.Length; i++)
        {
            var diff = _predictedValues[i] - _expectedValues[i];
            squaredSum += diff * diff;
        }

        return squaredSum / 2;
    }

    [Benchmark]
    public float CalculateMseLinq()
    {
        var squaredSum = _predictedValues
            .AsParallel()
            .Select((t, i) => t - _expectedValues[i])
            .Sum(diff => diff * diff);
        return squaredSum / 2;
    }

    [Benchmark]
    public float CalculateMsePlinq()
    {
        var squaredSum = _predictedValues
            .AsParallel()
            .Select((t, i) => t - _expectedValues[i])
            .Sum(diff => diff * diff);
        return squaredSum / 2;
    }
}
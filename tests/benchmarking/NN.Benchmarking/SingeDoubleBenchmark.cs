using BenchmarkDotNet.Attributes;

namespace NN.Benchmarking;

[SimpleJob]
public class SingeDoubleBenchmark
{
    [Benchmark(Baseline = true)]
    public float SingleAddition()
    {
        float a = 1.0f, b = 2.0f;
        return a + b;
    }

    [Benchmark]
    public double DoubleAddition()
    {
        double a = 1.0, b = 2.0;
        return a + b;
    }

    [Benchmark]
    public float SingleMultiplication()
    {
        float a = 1.0f, b = 2.0f;
        return a * b;
    }

    [Benchmark]
    public double DoubleMultiplication()
    {
        double a = 1.0, b = 2.0;
        return a * b;
    }
}
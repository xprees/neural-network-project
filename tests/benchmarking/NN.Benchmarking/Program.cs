using BenchmarkDotNet.Running;
using NN.Benchmarking.ActivationFunctions;

BenchmarkRunner.Run<SoftmaxActivationBenchmark>();
BenchmarkRunner.Run<SoftmaxDeriveBenchmark>();
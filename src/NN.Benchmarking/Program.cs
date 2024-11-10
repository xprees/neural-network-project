using BenchmarkDotNet.Running;
using NN.Benchmarking.LossFunctions;

BenchmarkRunner.Run<MeanSquaredErrorBenchmarks>();
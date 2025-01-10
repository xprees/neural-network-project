using NNProject.Networks;
using NNProject.Performance;

using var totalStopwatch = new DisposableStopwatch();
totalStopwatch.Start();

//(epochs: 50, batch size: 128, learning rate: 0.009, momentum: 0.4) - 85.15% acc (Sigmoid + Sgd momentum + CrossEntropy) 
//{ MaxEpochs = 50, BatchSize = 512, LearningRate = 0.0001, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 1 } - 85.89% Softmax + Adam + CrossEntropy
// { MaxEpochs = 50, BatchSize = 1024, LearningRate = 0.0003, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 42 } - 86.32% Soft + Adam + Cross - 784 -> 64 -> 32 -> 10
// { MaxEpochs = 50, BatchSize = 1024, LearningRate = 0.0002, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 42 } - 86.35% (same)
// { MaxEpochs = 50, BatchSize = 1024, LearningRate = 0.00025, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 42 } - 85.65% (same) -> 784 -> 128 -> 64 -> 10
// Parameters: MnistNnOptions { MaxEpochs = 25, BatchSize = 512, LearningRate = 0.00012, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 1, ShuffleData = True } - (19 ep) 87.29% (784 -> 256 -> 10)
// { MaxEpochs = 35, BatchSize = 256, LearningRate = 0.0001, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 1, ShuffleData = True }) - 87.76% after 22 epochs (sloow) - 784 -> 512 (drop 0.6) -> 64 (drop 0.2) -> 10

var mnistNn = new MnistNn(new MnistNnOptions(11, 512, 0.00017f, Seed: 1))
{
    Logging = true
};
var log = mnistNn.Run();

Console.WriteLine(log);

var totalRunTime = totalStopwatch.ElapsedMilliseconds;
Console.WriteLine($"Total time: {totalRunTime} ms ({totalRunTime / (1000f * 60):F} min)");
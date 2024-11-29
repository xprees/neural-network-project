using NNProject;
using NNProject.Performance;

using var totalStopwatch = new DisposableStopwatch();
totalStopwatch.Start();

//(epochs: 50, batch size: 128, learning rate: 0.009, momentum: 0.4) - 85.15% acc (Sigmoid + Sgd momentum + CrossEntropy) 
//{ MaxEpochs = 50, BatchSize = 512, LearningRate = 0.0001, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 1 } - 85.89% Softmax + Adam + CrossEntropy
// { MaxEpochs = 50, BatchSize = 1024, LearningRate = 0.0003, DecayRateOrBeta1 = 0.9, Beta2 = 0.999, Seed = 42 } - 86.32% Soft + Adam + Cross - 784 -> 64 -> 32 -> 10

var mnistNn = new MnistNn(new MnistNnOptions(50, 1024, 0.0003f, Seed: 42));
var nnStats = mnistNn.Run();

Console.WriteLine(nnStats);

var totalRunTime = totalStopwatch.ElapsedMilliseconds;
Console.WriteLine($"Total time: {totalRunTime} ms ({totalRunTime / (1000f * 60):F} min)");

// TODO implement saving best accuracy from all epochs and also killing
//  the training if the accuracy is not improving for a while
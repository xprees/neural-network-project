using NNProject;
using NNProject.Performance;

using var totalStopwatch = new DisposableStopwatch();
totalStopwatch.Start();

//(epochs: 50, batch size: 128, learning rate: 0.009, momentum: 0.4) - 85.15% acc (Sigmoid + Sgd momentum + CrossEntropy) 

var mnistNn = new MnistNn(new MnistNnOptions(50, 256, 0.0001f));
var nnStats = mnistNn.Run();

Console.WriteLine(nnStats);

var totalRunTime = totalStopwatch.ElapsedMilliseconds;
Console.WriteLine($"Total time: {totalRunTime} ms ({totalRunTime / (1000f * 60):F} min)");

// TODO implement saving best accuracy from all epochs and also killing
//  the training if the accuracy is not improving for a while
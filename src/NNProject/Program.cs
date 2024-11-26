using NNProject;
using NNProject.Performance;

using var totalStopwatch = new DisposableStopwatch();
totalStopwatch.Start();

//(epochs: 50, batch size: 64, learning rate: 0.009, momentum: 0.4) - 85.15% acc

var mnistNn = new MnistNn(50, 64, 0.009f, 0.4f);
var nnStats = mnistNn.Run();

Console.WriteLine(nnStats);

var totalRunTime = totalStopwatch.ElapsedMilliseconds;
Console.WriteLine($"Total time: {totalRunTime} ms ({totalRunTime / (1000f * 60):F} min)");
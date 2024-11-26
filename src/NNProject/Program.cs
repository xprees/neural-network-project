using NNProject;
using NNProject.Performance;

using var totalStopwatch = new DisposableStopwatch();
totalStopwatch.Start();

var mnistNn = new MnistNn(10, 64, 0.005f, 0);
var nnStats = mnistNn.Run();

Console.WriteLine(nnStats);

var totalRunTime = totalStopwatch.ElapsedMilliseconds;
Console.WriteLine($"Total time: {totalRunTime} ms ({totalRunTime / (1000f * 60):F} min)");
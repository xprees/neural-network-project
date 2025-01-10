using NNProject.Networks;
using NNProject.Performance;

using var totalStopwatch = new DisposableStopwatch();
totalStopwatch.Start();

var mnistNn = new MnistNn(new MnistNnOptions(11, 512, 0.00017f, Seed: 1))
{
    Logging = true
};
var log = mnistNn.Run();

Console.WriteLine(log);

var totalRunTime = totalStopwatch.ElapsedMilliseconds;
Console.WriteLine($"Total time: {totalRunTime} ms ({totalRunTime / (1000f * 60):F} min)");
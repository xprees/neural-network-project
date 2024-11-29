using DataProcessing.Evaluation.Metrics;
using NNProject.Networks;

namespace NNProject.Logging;

public record NnEpochLog(int Epoch, StatisticalMetrics Stats, float Accuracy, float Loss, long TimeTook)
{
    public void LogToConsole() => Console.WriteLine($"\t{ToString()}");

    public override string ToString() =>
        $"Epoch: {Epoch + 1:00} - Accuracy {Accuracy * 100:F2}% - Loss {Loss:F4} - Time {TimeTook} ms";
}

public record NnRunLog(List<NnEpochLog> EpochLogs, MnistNnOptions Options)
{
    public float AverageAccuracy => EpochLogs.Average(l => l.Accuracy);
    public float BestAccuracy => EpochLogs.Max(l => l.Accuracy);
    public float MinAccuracy => EpochLogs.Min(l => l.Accuracy);

    public StatisticalMetrics FinalMetrics { get; set; }

    public long TotalTimeTook { get; set; }

    public void AddLog(NnEpochLog log) => EpochLogs.Add(log);

    public void AddLogs(IEnumerable<NnEpochLog> logs) => EpochLogs.AddRange(logs);

    public void ExportToFile(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine(ToString());
        writer.Flush();
    }

    public override string ToString() =>
        "================ RUN ===============\n" +
        $"Parameters: {Options}\n" +
        $"Total Time Took: {TotalTimeTook} ms\n" +
        $"Average Accuracy: {AverageAccuracy * 100:F2}%\n" +
        $"Best Accuracy: {BestAccuracy * 100:F2}%\n" +
        $"Min Accuracy: {MinAccuracy * 100:F2}%\n" +
        $"Final Metrics: {FinalMetrics}\n" +
        "=====================================\n" +
        "Epoch Logs:\n" +
        string.Join("\n", EpochLogs.Select(l => $"\t{l}"));
}
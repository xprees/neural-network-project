using System.Diagnostics;

namespace NNProject.Performance;

public class DisposableStopwatch : IDisposable
{
    private readonly Stopwatch _stopwatch = new();
    public long ElapsedMilliseconds => _stopwatch.ElapsedMilliseconds;

    public void Start() => _stopwatch.Start();

    public void Stop() => _stopwatch.Stop();

    public void Restart() => _stopwatch.Restart();

    public void Dispose() => _stopwatch.Stop();
}
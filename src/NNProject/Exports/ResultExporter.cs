namespace NNProject.Exports;

public static class ResultExporter
{
    public static async Task ExportResultsAsCsvAsync<T>(string filePath, IEnumerable<T> results)
    {
        await using var writer = new StreamWriter(filePath);
        foreach (var result in results)
        {
            await writer.WriteLineAsync($"{result}");
        }
    }
}
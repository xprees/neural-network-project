namespace NNProject;

public static class DatasetPathFinder
{
    private static string? _dataFolderPath = null;

    public const string DatasetFolderName = "data";
    public const string TrainVectors = "fashion_mnist_train_vectors.csv";
    public const string TrainLabels = "fashion_mnist_train_labels.csv";
    public const string TestVectors = "fashion_mnist_test_vectors.csv";
    public const string TestLabels = "fashion_mnist_test_labels.csv";

    public static string? GetTrainVectorsPath()
    {
        if (_dataFolderPath is not null) return Path.Combine(_dataFolderPath, TrainVectors);

        if (!TryFindDataSetFolderPathAndCacheIt()) return null;

        return Path.Combine(_dataFolderPath!, TrainVectors);
    }

    public static string? GetTrainLabelsPath()
    {
        if (_dataFolderPath is not null) return Path.Combine(_dataFolderPath, TrainLabels);

        if (!TryFindDataSetFolderPathAndCacheIt()) return null;

        return Path.Combine(_dataFolderPath!, TrainLabels);
    }

    public static string? GetTestVectorsPath()
    {
        if (_dataFolderPath is not null) return Path.Combine(_dataFolderPath, TestVectors);

        if (!TryFindDataSetFolderPathAndCacheIt()) return null;

        return Path.Combine(_dataFolderPath!, TestVectors);
    }

    public static string? GetTestLabelsPath()
    {
        if (_dataFolderPath is not null) return Path.Combine(_dataFolderPath, TestLabels);

        if (!TryFindDataSetFolderPathAndCacheIt()) return null;

        return Path.Combine(_dataFolderPath!, TestLabels);
    }

    private static bool TryFindDataSetFolderPathAndCacheIt()
    {
        var solutionRoot = AppContext.BaseDirectory;
        while (solutionRoot != null && !Directory.Exists(Path.Combine(solutionRoot, DatasetFolderName)))
        {
            solutionRoot = Directory.GetParent(solutionRoot)?.FullName;
        }

        if (solutionRoot == null) return false;

        _dataFolderPath = Path.Combine(solutionRoot, DatasetFolderName);
        return true;
    }
}
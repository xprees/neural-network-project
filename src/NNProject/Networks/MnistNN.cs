using DataProcessing.Encoding;
using DataProcessing.Evaluation;
using DataProcessing.Evaluation.Metrics;
using DataProcessing.Loading;
using DataProcessing.Preprocessing;
using NNProject.Exports;
using NNProject.Logging;
using NNProject.Performance;
using NNStructure;
using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

namespace NNProject.Networks;

public class MnistNn(MnistNnOptions options)
{
    private const int ClassesCount = 10;

    private readonly string _rootRepoPath =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    private readonly OneHotEncoder<int> _oneHotEncoder = new(Enumerable.Range(0, ClassesCount));
    private readonly Preprocessing _preprocessing = new();
    private readonly MnistEvaluator _evaluator = new();

    private readonly int _maxEpochs = options.MaxEpochs;
    private readonly int _batchSize = options.BatchSize;
    private readonly float _learningRate = options.LearningRate;
    private readonly float _decayRateOrBeta1 = options.DecayRateOrBeta1;
    private readonly float _beta2 = options.Beta2;
    private readonly int _seed = options.Seed;
    private readonly bool _shuffleData = options.ShuffleData;

    // NN components
    private ILossFunction _lossFunction = null!;

    /// Enable or disable logging to console
    public bool Logging { get; set; } = true;

    private NeuralNetwork CreateNetwork()
    {
        _lossFunction = new CrossEntropy();
        var nn = new NeuralNetwork(
            _lossFunction,
            new GlorotWeightInitializer(_seed),
            new AdamW(_learningRate, _decayRateOrBeta1, _beta2, 0.001f),
            _seed
        );
        nn.AddLayer(new FullyConnectedDropoutLayer(784, 512, new Relu(), 0.6f));
        nn.AddLayer(new FullyConnectedDropoutLayer(512, 64, new Relu(), 0.2f));
        nn.AddLayer(new FullyConnectedLayer(64, ClassesCount, new Softmax()));
        // Make sure you are using Softmax in the output layer when using CrossEntropy loss function

        return nn;
    }

    private (float[][] trainInput, float[][] trainingExpectedOutput) PreprocessTrainingData(
        float[][] trainData, float[][] trainLabels)
    {
        using var stopwatch = new DisposableStopwatch();
        stopwatch.Start();
        if (Logging) Console.WriteLine("Preprocessing data...");

        var trainInput = _preprocessing
            .NormalizeByDivision(trainData)
            .ToArray();

        var trainingExpectedOutput = _oneHotEncoder
            .Encode(trainLabels.Select(x => (int)x.First()))
            .ToArray();

        var preprocessingTime = stopwatch.ElapsedMilliseconds;
        if (Logging) Console.WriteLine($"[DONE] Preprocessing data... Time: {preprocessingTime} ms");

        return (trainInput, trainingExpectedOutput);
    }

    private (float[][] result, StatisticalMetrics stats) TestNetwork(
        NeuralNetwork nn, float[][] testData, float[] testLabels
    )
    {
        var result = nn.Test(testData);

        var stats = _evaluator.EvaluateModel(result, testLabels);

        return (result, stats);
    }

    private void TrainNetwork(NeuralNetwork nn, float[][] trainingData, float[][] trainingLabels)
    {
        using var stopwatch = new DisposableStopwatch();
        nn.InitializeWeights();

        if (Logging) Console.WriteLine($"Training neural network... \n({options})");

        stopwatch.Restart();

        nn.Train(trainingData, trainingLabels, _maxEpochs, _batchSize, _shuffleData);

        var trainingTime = stopwatch.ElapsedMilliseconds;
        if (Logging) Console.WriteLine($"[DONE] Training neural network... Time: {trainingTime} ms");
    }

    public NnRunLog Run(string[]? args = null)
    {
        var runLog = new NnRunLog([], options);
        using var stopWatch = new DisposableStopwatch();
        stopWatch.Start();

        var (trainDataPath, trainLabelsPath, testDataPath, testLabelsPath) = GetDatasetFilesPaths(args ?? []);

        var (trainData, trainLabels) = LoadTrainingData(trainDataPath, trainLabelsPath);

        var (trainInput, trainingExpectedOutput) = PreprocessTrainingData(trainData, trainLabels);
        var trainLabels1D = trainLabels.Select(x => x.First()).ToArray();

        var nn = CreateNetwork();

        var (testData, testLabels, _) = LoadTestingData(testDataPath, testLabelsPath);

        var epochStopwatch = new DisposableStopwatch();
        nn.OnEpochEnd += (_, arg) =>
        {
            if (Logging)
            {
                Console.WriteLine($"\tEpoch {arg.Epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms");
            }

            epochStopwatch.Restart();
        };

        epochStopwatch.Start();
        TrainNetwork(nn, trainInput, trainingExpectedOutput);

        var (testResult, testStats) = TestNetwork(nn, testData, testLabels);
        runLog.FinalTestMetrics = testStats;

        var (trainResult, trainStats) = TestNetwork(nn, trainInput, trainLabels1D);
        runLog.FinalTrainMetrics = trainStats;

        runLog.TotalTimeTook = stopWatch.ElapsedMilliseconds;
        ExportResults(testResult, $"{_rootRepoPath}/test_predictions.csv");
        ExportResults(trainResult, $"{_rootRepoPath}/train_predictions.csv");

        return runLog;
    }

    #region Data handling

    private (float[][] trainData, float[][] trainLabels) LoadTrainingData(string trainDataPath, string trainLabelsPath)
    {
        using var stopwatch = new DisposableStopwatch();
        if (Logging) Console.WriteLine("Loading training data...");
        stopwatch.Start();

        using var trainDataLoader = new DataLoader(trainDataPath);
        var trainData = trainDataLoader.ReadAllVectors();

        using var trainLabelsLoader = new DataLoader(trainLabelsPath);
        var trainLabels = trainLabelsLoader.ReadAllVectors();

        var trainDataTime = stopwatch.ElapsedMilliseconds;
        if (Logging) Console.WriteLine($"[DONE] Loading training data... Time: {trainDataTime} ms");

        return (trainData, trainLabels);
    }

    private (float[][] testData, float[] testLabels, float[][] testLabelsOneHot) LoadTestingData(string testDataPath,
        string testLabelsPath)
    {
        using var stopwatch = new DisposableStopwatch();
        if (Logging) Console.WriteLine("Loading test data...");
        stopwatch.Restart();

        using var testDataLoader = new DataLoader(testDataPath);
        var testData = testDataLoader.ReadAllVectors();

        using var testLabelsLoader = new DataLoader(testLabelsPath);
        var testLabels = testLabelsLoader.ReadAllVectors()
            .Select(l => l.First())
            .ToArray();

        var testLabelsOneHot = _oneHotEncoder.Encode(testLabels.Select(x => (int)x)).ToArray();

        var testDataTime = stopwatch.ElapsedMilliseconds;
        if (Logging) Console.WriteLine($"[DONE] Loading test data... Time: {testDataTime} ms");

        return (testData, testLabels, testLabelsOneHot);
    }

    private void ExportResults(float[][] result, string path)
    {
        try
        {
            var decodedResult = _oneHotEncoder.Decode(result);
            ResultExporter.ExportResultsAsCsv(path, decodedResult);

            if (Logging) Console.WriteLine($"Results exported to {path}");
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"Error exporting results: {e.Message}");
        }
    }

    #endregion

    #region Dataset paths

    /// Parses the command line arguments to get the paths to the dataset files or tries to find them.
    private (string trainDataPath, string trainLabelsPath, string testDataPath, string testLabelsPath)
        GetDatasetFilesPaths(string[] args)
    {
        string? trainDataPath = null;
        var hasArgumentsWithDataPaths = args.Length >= 4;
        if (hasArgumentsWithDataPaths) trainDataPath = args[0];

        trainDataPath ??= DatasetPathFinder.GetTrainVectorsPath()
                          ?? throw new ArgumentException("No path for trainData provided and no default path found");

        string? trainLabelsPath = null;
        if (hasArgumentsWithDataPaths) trainLabelsPath = args[1];
        trainLabelsPath ??= DatasetPathFinder.GetTrainLabelsPath()
                            ?? throw new ArgumentException(
                                "No path for trainLabels provided and no default path found");

        string? testDataPath = null;
        if (hasArgumentsWithDataPaths) testDataPath = args[2];
        testDataPath ??= DatasetPathFinder.GetTestVectorsPath()
                         ?? throw new ArgumentException("No path for testData provided and no default path found");

        string? testLabelsPath = null;
        if (hasArgumentsWithDataPaths) testLabelsPath = args[3];
        testLabelsPath ??= DatasetPathFinder.GetTestLabelsPath()
                           ?? throw new ArgumentException("No path for testLabels provided and no default path found");

        return (trainDataPath, trainLabelsPath, testDataPath, testLabelsPath);
    }

    #endregion
}
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

    public bool SkipOnEpochTesting { get; set; } = false;

    private NeuralNetwork CreateNetwork()
    {
        _lossFunction = new CrossEntropy();
        var nn = new NeuralNetwork(
            _lossFunction,
            new GlorotWeightInitializer(_seed),
            new AdamW(_learningRate, _decayRateOrBeta1, _beta2, 0.001f),
            _seed
        );
        nn.AddLayer(new FullyConnectedDropoutLayer(784, 256, new Relu(), 0.45f));
        nn.AddLayer(new FullyConnectedLayer(256, ClassesCount, new Softmax()));
        // Make sure you are using Softmax in the output layer when using CrossEntropy loss function

        return nn;
    }

    private (float[][] data, float[][] dataLabelsOneHot) PreprocessData(float[][] inputData, float[][] inputLabels)
    {
        using var stopwatch = new DisposableStopwatch();
        stopwatch.Start();
        if (Logging) Console.WriteLine("Preprocessing data...");

        var processedData = _preprocessing
            .NormalizeByDivision(inputData)
            .ToArray();

        var processedLabels = _oneHotEncoder
            .Encode(inputLabels.Select(x => (int)x.First()))
            .ToArray();

        var preprocessingTime = stopwatch.ElapsedMilliseconds;
        if (Logging) Console.WriteLine($"[DONE] Preprocessing data... Time: {preprocessingTime} ms");

        return (processedData, processedLabels);
    }

    private (float[][] result, StatisticalMetrics stats) TestNetwork(
        NeuralNetwork nn, float[][] testData, float[] testLabels1D
    )
    {
        var result = nn.Test(testData);

        var stats = _evaluator.EvaluateModel(result, testLabels1D);

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

        var (
            trainData, trainLabelsOneHot,
            validationData, validationLabelsOneHot
            ) = GetTrainingAndValidationData(args ?? []);
        var trainLabels1D = trainLabelsOneHot.Select(x => x.First()).ToArray();
        var validationLabels1D = validationLabelsOneHot.Select(x => x.First()).ToArray();

        var nn = CreateNetwork();

        var epochStopwatch = new DisposableStopwatch();
        nn.OnEpochEnd += (_, arg) =>
        {
            if (SkipOnEpochTesting)
            {
                if (Logging)
                {
                    Console.WriteLine($"\tEpoch {arg.Epoch + 1} completed in {epochStopwatch.ElapsedMilliseconds} ms");
                }

                epochStopwatch.Restart();

                return;
            }

            var (validationEpochResult, validationEpochStats) = TestNetwork(nn, validationData, validationLabels1D);

            // Validation
            var validationLoss = validationEpochResult.Zip(validationLabelsOneHot,
                    (predicted, expected) => _lossFunction.Calculate(predicted, expected))
                .Average();

            var epochTime = epochStopwatch.ElapsedMilliseconds;
            var epochLog = new NnEpochLog(arg.Epoch, validationEpochStats, validationEpochStats.Accuracy,
                validationLoss,
                epochTime);
            runLog.AddLog(epochLog);

            if (Logging)
            {
                epochLog.LogToConsole();
            }

            epochStopwatch.Restart();
        };

        epochStopwatch.Start();
        TrainNetwork(nn, trainData, trainLabelsOneHot);

        // Evaluate Test Data
        var (testData, testLabelsOneHot) = GetTestingData(args ?? []);
        var testLabels1D = testLabelsOneHot.Select(x => x.First()).ToArray();

        var (testResult, testStats) = TestNetwork(nn, testData, testLabels1D);
        runLog.FinalTestMetrics = testStats;

        // Evaluate Training Data
        var (trainResult, trainStats) = TestNetwork(nn, trainData, trainLabels1D);
        runLog.FinalTrainMetrics = trainStats;

        runLog.TotalTimeTook = stopWatch.ElapsedMilliseconds;
        ExportResults(testResult, $"{_rootRepoPath}/test_predictions.csv");
        ExportResults(trainResult, $"{_rootRepoPath}/train_predictions.csv");

        return runLog;
    }

    #region Data handling

    private (float[][] trainData, float[][] trainLabelsOneHot, float[][] validationData, float[][]validationLabelsOneHot
        )
        GetTrainingAndValidationData(string[] args)
    {
        var (trainDataPath, trainLabelsPath, _, _) = GetDatasetFilesPaths(args);
        var (inputTrainData, inputTrainLabels) = LoadTrainingData(trainDataPath, trainLabelsPath);
        var (trainingData, trainingExpectedOutput) = PreprocessData(inputTrainData, inputTrainLabels); // TODO fix

        // Split data into 80% training and 20% validation 
        var validationDataCount = (int)(trainingData.Length * 0.2f);

        var validationData = trainingData.Take(validationDataCount).ToArray();
        var validationLabelsOneHot = trainingExpectedOutput.Take(validationDataCount).ToArray();

        var trainData = trainingData.Skip(validationDataCount).ToArray();
        var trainLabelsOneHot = trainingExpectedOutput.Skip(validationDataCount).ToArray();

        return (trainData, trainLabelsOneHot, validationData, validationLabelsOneHot);
    }

    private (float[][] testData, float[][] testLabelsOneHot) GetTestingData(string[] args)
    {
        var (_, _, testDataPath, testLabelsPath) = GetDatasetFilesPaths(args);
        var (rawTestData, rawTestLabelsOneHot) = LoadTestingData(testDataPath, testLabelsPath);

        var (testData, testLabelsOneHotProcessed) = PreprocessData(rawTestData, rawTestLabelsOneHot); // TODO fix

        return (testData, testLabelsOneHotProcessed);
    }

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

    private (float[][] testData, float[][] testLabelsOneHot) LoadTestingData(string testDataPath,
        string testLabelsPath)
    {
        using var stopwatch = new DisposableStopwatch();
        if (Logging) Console.WriteLine("Loading test data...");
        stopwatch.Restart();

        using var testDataLoader = new DataLoader(testDataPath);
        var testData = testDataLoader.ReadAllVectors();

        using var testLabelsLoader = new DataLoader(testLabelsPath);
        var testLabels = testLabelsLoader.ReadAllVectors()
            .Select(l => l.First()) // Load first scalar of vector
            .Select(x => (int)x); // Runtime float -> int cast 

        var testLabelsOneHot = _oneHotEncoder.Encode(testLabels).ToArray();

        var testDataTime = stopwatch.ElapsedMilliseconds;
        if (Logging) Console.WriteLine($"[DONE] Loading test data... Time: {testDataTime} ms");

        return (testData, testLabelsOneHot);
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
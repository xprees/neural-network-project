using DataProcessing.Encoding;
using DataProcessing.Evaluation;
using DataProcessing.Loading;
using DataProcessing.Preprocessing;
using NNProject;
using NNProject.Exports;
using NNProject.Performance;
using NNStructure;
using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

using var totalStopwatch = new DisposableStopwatch();
totalStopwatch.Start();
using var stopwatch = new DisposableStopwatch();

// Load the training data
Console.WriteLine("Loading training data...");
stopwatch.Start();

var (trainDataPath, trainLabelsPath, testDataPath, testLabelsPath) = GetDatasetFilesPaths();

using var trainDataLoader = new DataLoader(trainDataPath);
var trainData = trainDataLoader.ReadAllVectors();

using var trainLabelsLoader = new DataLoader(trainLabelsPath);
var trainLabels = trainLabelsLoader.ReadAllVectors();

var trainDataTime = stopwatch.ElapsedMilliseconds;
Console.WriteLine($"[DONE] Loading training data... Time: {trainDataTime} ms");

// Preprocess the data
Console.WriteLine("Preprocessing data...");
stopwatch.Restart();

var preprocessing = new Preprocessing();
var normalizedData = preprocessing.NormalizeByDivision(trainData);

var oneHotEncoder = new OneHotEncoder<int>(Enumerable.Range(0, 10));

var trainLabelsOneHot = oneHotEncoder.Encode(trainLabels.Select(x => (int)x.First()));

var (trainInput, trainingExpectedOutput) =
    preprocessing.ShuffleData(normalizedData.ToArray(), trainLabelsOneHot.ToArray());

var preprocessingTime = stopwatch.ElapsedMilliseconds;
Console.WriteLine($"[DONE] Preprocessing data... Time: {preprocessingTime} ms");

// Create the neural network
var nn = new NeuralNetwork(new MeanSquaredError(), new GlorotWeightInitializer(0), new SgdOptimizer(0.2f));
nn.AddLayer(new FullyConnectedLayer(784, 256, new Relu()));
nn.AddLayer(new FullyConnectedLayer(256, 64, new Relu()));
nn.AddLayer(new FullyConnectedLayer(64, 10, new Relu())); // TODO: Change to Softmax

nn.InitializeWeights();

// Train the neural network
Console.WriteLine("Training neural network...");
stopwatch.Restart();

nn.Train(trainInput, trainingExpectedOutput, 30, 64);

var trainingTime = stopwatch.ElapsedMilliseconds;
Console.WriteLine($"[DONE] Training neural network... Time: {trainingTime} ms");

Console.WriteLine("Loading test data...");
stopwatch.Restart();

using var testDataLoader = new DataLoader(testDataPath);
var testData = testDataLoader.ReadAllVectors();

using var testLabelsLoader = new DataLoader(testLabelsPath);
var testLabels = testLabelsLoader.ReadAllVectors()
    .Select(l => l.First())
    .ToArray();

var testDataTime = stopwatch.ElapsedMilliseconds;
Console.WriteLine($"[DONE] Loading test data... Time: {testDataTime} ms");

Console.WriteLine("Testing neural network...");

var result = nn.Test(testData);

var testingTime = stopwatch.ElapsedMilliseconds;
Console.WriteLine($"[DONE] Testing neural network... Time: {testingTime} ms");

Console.WriteLine("Evaluating accuracy...");
var evaluator = new MnistEvaluator();
var stats = evaluator.EvaluateModel(result, testLabels);

var evalTime = stopwatch.ElapsedMilliseconds;
Console.WriteLine($"Accuracy: {stats.Accuracy:F}");
Console.WriteLine($"Precision: {stats.Precision:F}");
Console.WriteLine($"Recall: {stats.Recall:F}");
Console.WriteLine($"F1 Score: {stats.F1Score:F}");

Console.WriteLine($"[DONE] Evaluating accuracy... Time: {evalTime} ms");

var decodedResult = oneHotEncoder.Decode(result);
ResultExporter.ExportResultsAsCsv("./results.csv", decodedResult);

Console.WriteLine("Results exported to results.csv");

var totalTime = totalStopwatch.ElapsedMilliseconds;
Console.WriteLine($"Total time: {totalTime} ms ({totalTime / (1000f * 60):F} min)");

return;

// Args: trainDataPath trainLabelsPath testDataPath testLabelsPath
(string trainDataPath, string trainLabelsPath, string testDataPath, string testLabelsPath) GetDatasetFilesPaths()
{
    string? trainDataPath = null;
    var hasArgumentsWithDataPaths = args.Length >= 4;
    if (hasArgumentsWithDataPaths) trainDataPath = args[0];

    trainDataPath ??= DatasetPathFinder.GetTrainVectorsPath()
                      ?? throw new ArgumentException("No path for trainData provided and no default path found");

    string? trainLabelsPath = null;
    if (hasArgumentsWithDataPaths) trainLabelsPath = args[1];
    trainLabelsPath ??= DatasetPathFinder.GetTrainLabelsPath()
                        ?? throw new ArgumentException("No path for trainLabels provided and no default path found");

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
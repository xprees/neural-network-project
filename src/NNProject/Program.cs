using DataProcessing.Loading;
using NNProject;
using NNStructure;
using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

// Load the data
string? trainDataPath = null;
if (args.Length > 0) trainDataPath = args[0];

trainDataPath ??= DatasetPathFinder.GetTestVectorsPath()
                  ?? throw new ArgumentException("No path provided and no default path found");

using var dataLoader = new DataLoader(trainDataPath);


// Preprocess the data

// Create the neural network
var nn = new NeuralNetwork(new MeanSquaredError(), new GlorotWeightInitializer(), new SgdOptimizer(0.8f));
nn.AddLayer(new FullyConnectedLayer(2, 2, new Tanh()));
nn.AddLayer(new FullyConnectedLayer(2, 1, new Tanh()));

nn.InitializeWeights();

float[][] inputs =
[
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
];

float[][] expectedResults =
[
    [0],
    [1],
    [1],
    [0]
];

nn.Train(inputs, expectedResults, 100, 4);

var result = nn.ForwardPropagate([1, 1]).prediction.FirstOrDefault();

Console.WriteLine($"For [1, 1] expected 0, got {result}");
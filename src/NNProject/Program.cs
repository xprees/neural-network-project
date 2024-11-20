using NNStructure;
using NNStructure.ActivationFunctions;
using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;
using NNStructure.Optimizers;

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
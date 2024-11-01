using NNStructure.Initialization;
using NNStructure.Layers;
using NNStructure.LossFunctions;

namespace NNStructure;

public class NeuralNetwork(ILossFunction lossFunction, IWeightsInitializer initializer)
{
    public List<ILayer> Layers { get; } = [];

    public void AddLayer(ILayer layer) => Layers.Add(layer);

    public void InitializeWeights()
    {
        foreach (var layer in Layers)
        {
            layer.InitializeWeights(initializer);
        }
    }

    /// Does the forward propagation for the input vector and returns prediction vector 
    public float[] ForwardPropagate(float[] input)
    {
        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.DoForwardPass(output);
        }

        return output;
    }

    public void BackPropagate(float[] predictedOutput, float[] expectedOutput)
    {
        var lossGradient = lossFunction.CalculateGradient(predictedOutput, expectedOutput);

        for (var i = Layers.Count - 1; i >= 0; i--)
        {
            var layer = Layers[i];
            lossGradient = layer.DoBackpropagation(lossGradient);
        }
    }


    public void Train(float[][] inputMiniBatch, float[][] expectedResults, int epochs, float learningRate)
    {
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            for (var i = 0; i < inputMiniBatch.Length; i++)
            {
                var miniBatch = inputMiniBatch[i];
                var predictedOutput = ForwardPropagate(miniBatch);

                BackPropagate(predictedOutput, expectedResults[i]);

                foreach (var layer in Layers)
                {
                    layer.UpdateWeights(learningRate);
                }
            }
        }
    }
}
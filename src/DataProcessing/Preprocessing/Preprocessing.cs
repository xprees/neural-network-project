namespace DataProcessing.Preprocessing;

public class Preprocessing
{
    public float[][] NormalizeByDivision(float[][] input)
    {
        return input
            .Select(subArray => subArray.Select(element => element / 255).ToArray())
            .ToArray();
    }

    /// Input is intended for the training data and output for the labels  
    public (float[][] trainInput, float[][] expectedOutput) ShuffleData(float[][] input, float[][] output,
        int seed = 42)
    {
        if (input.Length != output.Length)
        {
            throw new ArgumentException("Input and output arrays must have the same length");
        }

        var random = new Random(seed);
        var shuffledIndices = Enumerable.Range(0, input.Length).ToArray();
        random.Shuffle(shuffledIndices);

        var shuffledInput = new float[input.Length][];
        var shuffledOutput = new float[output.Length][];
        foreach (var index in shuffledIndices)
        {
            shuffledInput[index] = input[index];
            shuffledOutput[index] = output[index];
        }

        return (shuffledInput, shuffledOutput);
    }
}
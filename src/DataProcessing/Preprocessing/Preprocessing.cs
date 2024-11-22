namespace DataProcessing.Preprocessing;

public class Preprocessing
{
    public float[][] NormalizeByDivision(float[][] input)
    {
        return input
            .Select(subArray => subArray.Select(element => element / 255).ToArray())
            .ToArray();
    }
}
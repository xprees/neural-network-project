namespace NNStructure.Initialization;

public static class RandomInNormalDistribution
{
    public static float GetRandomInNormalDistribution(Random random, float standardDeviation)
    {
        var x1 = 1 - random.NextSingle();
        var x2 = 1 - random.NextSingle();

        var y1 = (float)(Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2));
        return y1 * standardDeviation;
    }
}
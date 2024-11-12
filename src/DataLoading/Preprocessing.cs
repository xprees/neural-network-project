namespace DataLoading;

public static class Preprocessing
{
    public static float[]?[] NormalizeByDivision(float[]?[] input)
    {
        return input
            .Select(subArray => subArray?.Select(element => element / 256).ToArray())
            .ToArray();
    }

}
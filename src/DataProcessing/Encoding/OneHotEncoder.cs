namespace DataProcessing.Encoding;

/// One-hot encoder for encoding categorical data into one-hot vectors
public class OneHotEncoder<TLabel> where TLabel : notnull
{
    private readonly Dictionary<TLabel, int> _categoryMap;
    private readonly int _categoriesMap;

    public OneHotEncoder(IEnumerable<TLabel> labels)
    {
        _categoryMap =
            labels
                .Distinct()
                .Select((label, index) => new { label, index })
                .ToDictionary(x => x.label, x => x.index);
        _categoriesMap = _categoryMap.Count;
    }

    public IEnumerable<float[]> Encode(IEnumerable<TLabel> labels) => labels.Select(Encode);

    public float[] Encode(TLabel label)
    {
        if (!_categoryMap.TryGetValue(label, out var index))
            throw new ArgumentException($"Label '{label}' not found in the encoder.");

        var oneHotVector = new float[_categoriesMap];
        oneHotVector[index] = 1;
        return oneHotVector;
    }

    public IEnumerable<TLabel> Decode(IEnumerable<float[]> results) =>
        results
            .AsParallel()
            .AsOrdered()
            .Select(Decode);

    public TLabel Decode(float[] result)
    {
        if (result.Length != _categoriesMap)
        {
            throw new ArgumentException("The length of the result vector does not match the number of categories.");
        }

        var maxIndex = Array.IndexOf(result, result.Max());
        return _categoryMap
            .FirstOrDefault(x => x.Value == maxIndex)
            .Key;
    }
}
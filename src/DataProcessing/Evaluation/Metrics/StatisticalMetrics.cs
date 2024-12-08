namespace DataProcessing.Evaluation.Metrics;

public struct StatisticalMetrics()
{
    private const int ClassesCount = 10;
    public int[] TruePositives { get; } = new int[ClassesCount];
    public int[] TrueNegatives { get; } = new int[ClassesCount];
    public int[] FalsePositives { get; } = new int[ClassesCount];
    public int[] FalseNegatives { get; } = new int[ClassesCount];

    public float[] Precisions { get; } = new float[ClassesCount];
    public float[] Accuracies { get; } = new float[ClassesCount];
    public float[] Recalls { get; } = new float[ClassesCount];
    public float[] F1Scores { get; } = new float[ClassesCount];

    public float Precision { get; set; } = 0f;
    public float Accuracy { get; set; } = 0f;
    public float Recall { get; set; } = 0f;
    public float F1Score { get; set; } = 0f;

    public void FillMetric(int classIndex, int tp, int tn, int fp, int fn)
    {
        TruePositives[classIndex] = tp;
        TrueNegatives[classIndex] = tn;
        FalseNegatives[classIndex] = fn;
        FalsePositives[classIndex] = fp;
    }

    public void ComputeMetrics()
    {
        var totalPredictions = TruePositives[0] + TrueNegatives[0] + FalsePositives[0] + FalseNegatives[0];

        for (var i = 0; i < 10; i++)
        {
            Accuracies[i] = (float)(TruePositives[i] + TrueNegatives[i]) / totalPredictions;
            Precisions[i] = (float)TruePositives[i] / (TruePositives[i] + FalsePositives[i]);
            Recalls[i] = (float)TruePositives[i] / (TruePositives[i] + FalseNegatives[i]);
            F1Scores[i] = 2 * (Precisions[i] * Recalls[i]) / (Precisions[i] + Recalls[i]);
        }

        Precision = Precisions.Average();
        Recall = Recalls.Average();
        F1Score = F1Scores.Average();
        Accuracy = (float)TruePositives.Sum() / totalPredictions;
    }

    public override string ToString() =>
        $"Accuracy: {Accuracy * 100:2F}\nPrecision: {Precision:F}\nRecall: {Recall:F}\nF1 Score: {F1Score:F}";
}
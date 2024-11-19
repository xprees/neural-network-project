namespace DataLoading;

public struct StatisticalMetrics
{
    public int[] TruePositives;
    public int[] TrueNegatives;
    public int[] FalsePositives;
    public int[] FalseNegatives;

    public float[] Precisions;
    public float[] Accuracies;
    public float[] Recalls;
    public float[] F1Scores;

    public float Precision;
    public float Accuracy;
    public float Recall;
    public float F1Score;

    public StatisticalMetrics()
    {
        TruePositives = new int[10];
        TrueNegatives = new int[10];
        FalsePositives = new int[10];
        FalseNegatives = new int[10];
        
        Precisions = new float[10];
        Accuracies = new float[10];
        Recalls = new float[10];
        F1Scores = new float[10];

        Precision = 0;
        Accuracy = 0;
        Recall = 0;
        F1Score = 0;
    }
    

    public void FillMetrics(int classIndex, int tp, int tn, int fp, int fn)
    {
        TruePositives[classIndex] = tp;
        TrueNegatives[classIndex] = tn;
        FalseNegatives[classIndex] = fn;
        FalsePositives[classIndex] = fp;

        int totalPredictions = TruePositives[0] + TrueNegatives[0] + FalsePositives[0] + FalseNegatives[0];

        for (int i = 0; i < 10; i++)
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
}
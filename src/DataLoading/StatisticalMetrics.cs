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

    public void AssignConfusionMatrixCoefficients(int classIndex, int tp, int tn, int fp, int fn)
    {
        TruePositives[classIndex] = tp;
        TrueNegatives[classIndex] = tn;
        FalseNegatives[classIndex] = fn;
        FalsePositives[classIndex] = fp;
    }
}
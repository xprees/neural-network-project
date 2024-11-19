namespace DataLoading;

public struct StatisticalMetrics
{
    public int[] TruePositives { get;}
    public int[] TrueNegatives { get;}
    public int[] FalsePositives { get;}
    public int[] FalseNegatives { get;}

    public float[] Precisions { get;}
    public float[] Accuracies { get;}
    public float[] Recalls { get;}
    public float[] F1Scores { get;}

    public float Precision { get; set; }
    public float Accuracy { get; set; }
    public float Recall { get; set; }
    public float F1Score { get; set; }

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
    

    public void FillMetric(int classIndex, int tp, int tn, int fp, int fn)
    {
        TruePositives[classIndex] = tp;
        TrueNegatives[classIndex] = tn;
        FalseNegatives[classIndex] = fn;
        FalsePositives[classIndex] = fp;
    }

    public void ComputeMetrics()
    {
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
using DataLoading;
using FluentAssertions;

namespace DataLoadingTests;

[TestFixture]
public class StatisticalMetricsTests
{
    private StatisticalMetrics _metrics;

    private readonly int[][] _confusionMatrixData =
    [
        [60, 16, 17, 7],
        [95, 4, 1, 0],
        [35, 21, 15, 29],
        [82, 12, 3, 3],
        [37, 13, 45, 5],
        [99, 1, 0, 0],
        [37, 18, 15, 30],
        [52, 27, 20, 1],
        [26, 23, 48, 3],
        [29, 20, 31, 20],
    ];

    [SetUp]
    public void Setup()
    {
        _metrics = new StatisticalMetrics();

        for (var i = 0; i < _confusionMatrixData.Length; i++)
        {
            _metrics.FillMetric(
                i,
                _confusionMatrixData[i][0],
                _confusionMatrixData[i][1],
                _confusionMatrixData[i][2],
                _confusionMatrixData[i][3]);
        }
    }

    [Test]
    public void FillMetricTest()
    {
        var tp = _metrics.TruePositives;
        var tn = _metrics.TrueNegatives;
        var fn = _metrics.FalseNegatives;
        var fp = _metrics.FalsePositives;

        for (var i = 0; i < tp.Length; i++)
        {
            Assert.That(tp[i], Is.EqualTo(_confusionMatrixData[i][0]));
            Assert.That(tn[i], Is.EqualTo(_confusionMatrixData[i][1]));
            Assert.That(fp[i], Is.EqualTo(_confusionMatrixData[i][2]));
            Assert.That(fn[i], Is.EqualTo(_confusionMatrixData[i][3]));
        }
    }

    [Test]
    public void ComputeMetricsTest()
    {
        _metrics.ComputeMetrics();

        var accuracies = _metrics.Accuracies;
        var precisions = _metrics.Precisions;
        var recalls = _metrics.Recalls;
        var f1Scores = _metrics.F1Scores;

        float[] expAccuracies = [0.76f, 0.99f, 0.56f, 0.94f, 0.50f, 1.00f, 0.55f, 0.79f, 0.49f, 0.49f];
        float[] expPrecisions =
        [
            0.779221f, 0.989583f, 0.700000f, 0.964706f, 0.451220f, 1.000000f, 0.711538f, 0.722222f, 0.351351f, 0.483333f
        ];
        float[] expRecalls =
        [
            0.895522f, 1.000000f, 0.546875f, 0.964706f, 0.880952f, 1.000000f, 0.552239f, 0.981132f, 0.896552f, 0.591837f
        ];
        float[] expF1Scores =
        [
            0.833333f, 0.994764f, 0.614035f, 0.964706f, 0.596774f, 1.000000f, 0.621849f, 0.832000f, 0.504854f, 0.532110f
        ];

        for (var i = 0; i < accuracies.Length; i++)
        {
            accuracies[i].Should().BeApproximately(expAccuracies[i], 0.00001f);
            precisions[i].Should().BeApproximately(expPrecisions[i], 0.00001f);
            recalls[i].Should().BeApproximately(expRecalls[i], 0.00001f);
            f1Scores[i].Should().BeApproximately(expF1Scores[i], 0.00001f);
        }
    }
}
using DataLoading;
using FluentAssertions;

namespace DataLoadingTests;

[TestFixture]
public class MnistEvaluatorTests
{
    private MnistEvaluator _evaluator;

    #region Data

    private float[][] _dataPredicted =
    [
        [0.9936f, 0.8637f, 0.3062f, 0.8823f, 0.0478f, 0.8074f, 0.5804f, 0.9932f, 0.4565f, 0.7081f],
        [0.8536f, 0.5619f, 0.5813f, 0.2461f, 0.6229f, 0.0787f, 0.5705f, 0.0641f, 0.0417f, 0.7184f],
        [0.0492f, 0.4618f, 0.5718f, 0.8096f, 0.0841f, 0.8132f, 0.0657f, 0.7513f, 0.7043f, 0.8074f],
        [0.7446f, 0.1420f, 0.1657f, 0.4113f, 0.3758f, 0.4568f, 0.9141f, 0.3010f, 0.2309f, 0.6035f],
        [0.3735f, 0.9650f, 0.6697f, 0.9916f, 0.7935f, 0.7988f, 0.4908f, 0.4935f, 0.0194f, 0.0103f],
        [0.9297f, 0.3979f, 0.9859f, 0.5444f, 0.1250f, 0.7849f, 0.2258f, 0.0392f, 0.8500f, 0.6778f],
        [0.1064f, 0.4628f, 0.6199f, 0.7188f, 0.8610f, 0.0974f, 0.5186f, 0.6590f, 0.7066f, 0.7733f],
        [0.4233f, 0.8321f, 0.1770f, 0.8957f, 0.9304f, 0.8675f, 0.7434f, 0.3036f, 0.1579f, 0.9222f],
        [0.0567f, 0.7127f, 0.5740f, 0.2230f, 0.3472f, 0.3042f, 0.6567f, 0.0096f, 0.7086f, 0.0454f],
        [0.3130f, 0.0627f, 0.4765f, 0.4948f, 0.2469f, 0.6247f, 0.9418f, 0.4320f, 0.7987f, 0.9277f]
    ];

    private float[] _dataActual = [0f, 0f, 5f, 6f, 3f, 2f, 4f, 4f, 2f, 3f];

    #endregion

    [SetUp]
    public void Setup()
    {
        _evaluator = new MnistEvaluator();
    }

    [Test]
    public void EvaluateCompleteResultTest()
    {
        var result = new bool[10];
        for (var index = 0; index < _dataActual.Length; index++)
        {
            result[index] = _evaluator.EvaluateCompleteResult(_dataPredicted[index], (int)_dataActual[index]);
        }

        bool[] actualResult = [true, true, true, true, true, true, true, true, false, false];
        for (var index = 0; index < _dataActual.Length; index++)
        {
            Assert.That(result[index], Is.EqualTo(actualResult[index]));
        }
    }

    [Test]
    public void EvaluateCompletedResultsTest()
    {
        bool[] actualResult = [true, true, true, true, true, true, true, true, false, false];
        var result = _evaluator.EvaluateCompletedResults(_dataPredicted, _dataActual);
        for (var index = 0; index < _dataActual.Length; index++)
        {
            Assert.That(result[index], Is.EqualTo(actualResult[index]));
        }
    }

    [Test]
    public void ConvertToClassesTest()
    {
        int[] actualResult = [0, 0, 5, 6, 3, 2, 4, 4, 1, 6];
        var result = _evaluator.ConvertToClasses(_dataPredicted);
        for (var index = 0; index < _dataActual.Length; index++)
        {
            Assert.That(result[index] == actualResult[index]);
        }
    }

    [Test]
    public void EvaluateModelTest()
    {
        _dataPredicted =
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ];

        _dataActual = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        const int expectedFp = 1;
        const int expectedFn = 1;
        const int expectedTp = 1;
        const int expectedTn = 17;

        const float error = 0.00001f;
        const float expectedAccuracies = 0.9f;
        const float expectedPrecisions = 0.5f;
        const float expectedRecalls = 0.5f;
        const float expectedF1Scores = 0.5f;

        var metrics = _evaluator.EvaluateModel(_dataPredicted, _dataActual);

        for (var i = 0; i < 10; i++)
        {
            Assert.That(metrics.TruePositives[i] == expectedTp);
            Assert.That(metrics.TrueNegatives[i] == expectedTn);
            Assert.That(metrics.FalsePositives[i] == expectedFp);
            Assert.That(metrics.FalseNegatives[i] == expectedFn);

            expectedAccuracies.Should().BeApproximately(metrics.Accuracies[i], error);
            expectedPrecisions.Should().BeApproximately(metrics.Precisions[i], error);
            expectedRecalls.Should().BeApproximately(metrics.Recalls[i], error);
            expectedF1Scores.Should().BeApproximately(metrics.F1Scores[i], error);
        }

        metrics.Accuracy.Should().BeApproximately(0.5f, error);
        metrics.Precision.Should().BeApproximately(expectedPrecisions, error);
        metrics.Recall.Should().BeApproximately(expectedRecalls, error);
        metrics.F1Score.Should().BeApproximately(expectedF1Scores, error);
    }
}
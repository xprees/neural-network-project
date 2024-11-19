using DataLoading;

namespace DataLoadingTests;

public class UnitTestsPreprocessing
{
    [Test]
    public void PreprocessingTest()
    {
        Preprocessing preprocessing = new Preprocessing();
        float[][] data =
        [
            [1, 2, 3, 4, 5, 6],
            [255, 254, 243, 252, 251, 0]
        ];
        float[][] dataExpected =[
            [0.00392156862f, 0.007843133725f, 0.01176470588f, 0.01568627451f, 0.01960784314f, 0.02352941176f],
            [1.0f, 0.9960784314f, 0.9529411765f, 0.9882352941f, 0.9843137255f, 0.0f]
        ];
        
        data = preprocessing.NormalizeByDivision(data);

        for (int i = 0; i < data.Length; i++)
        {
            for (int j = 0; j < data[i].Length; j++)
            {
                Assert.Less(MathF.Abs(data[i][j] - dataExpected[i][j]), 0.00000001f);
            }
        }
    }
}
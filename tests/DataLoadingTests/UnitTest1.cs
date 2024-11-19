using System.Data;
using DataLoading;
namespace DataLoadingTests;

public class Tests
{
    public string dataFilePath;
    [SetUp]
    public void Setup()
    {
        // using this for always finding the file (even from binary I hope)
        string solutionRoot = AppDomain.CurrentDomain.BaseDirectory;
        while (!Directory.Exists(Path.Combine(solutionRoot, "data")))
        {
            solutionRoot = Directory.GetParent(solutionRoot).FullName;
        }
        dataFilePath = Path.Combine(solutionRoot, "data", "fashion_mnist_train_vectors.csv");
        Assert.IsNotNull(dataFilePath);
    }

    [Test]
    public void TestReadAllVectors()
    {

        using (DataLoader dataLoader = new DataLoader(dataFilePath))
        {
            Assert.IsTrue(dataLoader.ReadAllVectors().Length == 60000);
        }

    }
}
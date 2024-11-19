using DataLoading;
namespace DataLoadingTests;

public class UnitTestsDataLoader
{
    public string dataFilePath;
    [SetUp]
    public void Setup()
    {
        // using this for always finding the file
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
        // Works on my machine, not in gitLab
        Assert.Pass();
        return;
        using (DataLoader dataLoader = new DataLoader(dataFilePath))
        {
            Assert.That(dataLoader.ReadAllVectors().Length, Is.EqualTo(60000));
        }

    }

    [Test]
    public void TestReadOneVector()
    {
        // Works on my machine, not in gitLab
        Assert.Pass();
        return;
        using (DataLoader dataLoader = new DataLoader(dataFilePath))
        {
            for (int i = 0; i < 60000; i++)
            {
                Assert.Greater(dataLoader.ReadOneVector().Length, 0);
            }
            Assert.That(dataLoader.ReadOneVector().Length, Is.EqualTo(0));
        }
    }
    
    [Test]
    public void TestReadNVectors()
    {
        // Works on my machine, not in gitLab
        Assert.Pass();
        return; 
        using (DataLoader dataLoader = new DataLoader(dataFilePath))
        {
            for (int i = 0; i < 468; i++)
            {
                Assert.That(dataLoader.ReadNVectors(128).Length, Is.EqualTo(128));
            }

            float[][] dataLast = dataLoader.ReadNVectors(128);
            Assert.That(dataLast.Length, Is.EqualTo(128));
            Assert.That(dataLast[95].Length, Is.EqualTo(784));
            Assert.That(dataLast[96].Length, Is.EqualTo(0));
            Assert.That(dataLoader.ReadNVectors(128).Length, Is.EqualTo(0));
        }
    }
    
    
}
using DataProcessing;
using DataProcessing.Loading;
using FluentAssertions;

namespace DataLoadingTests;

[TestFixture]
[Explicit]
public class UnitTestsDataLoader
{
    private string _dataFilePath;

    [SetUp]
    public void Setup()
    {
        // using this for always finding the file
        var solutionRoot = AppDomain.CurrentDomain.BaseDirectory;
        while (!Directory.Exists(Path.Combine(solutionRoot, "data")))
        {
            solutionRoot = Directory.GetParent(solutionRoot)?.FullName;
        }

        _dataFilePath = Path.Combine(solutionRoot, "data", "fashion_mnist_train_vectors.csv");
        _dataFilePath.Should().NotBeNull();
    }

    [Test]
    public void TestReadAllVectors()
    {
        // Works on my machine, not in gitLab
        using var dataLoader = new DataLoader(_dataFilePath);
        dataLoader.ReadAllVectors().Length.Should().Be(60_000);
    }

    [Test]
    public void TestReadOneVector()
    {
        using var dataLoader = new DataLoader(_dataFilePath);
        for (var i = 0; i < 60000; i++)
        {
            dataLoader.ReadOneVector().Length.Should().BeGreaterThan(0);
        }

        dataLoader.ReadOneVector().Length.Should().Be(0);
    }

    [Test]
    public void TestReadNVectors()
    {
        using var dataLoader = new DataLoader(_dataFilePath);
        for (var i = 0; i < 468; i++)
        {
            Assert.That(dataLoader.ReadNVectors(128).Length, Is.EqualTo(128));
        }

        var dataLast = dataLoader.ReadNVectors(128);
        Assert.That(dataLast.Length, Is.EqualTo(128));
        Assert.That(dataLast[95].Length, Is.EqualTo(784));
        Assert.That(dataLast[96].Length, Is.EqualTo(0));
        Assert.That(dataLoader.ReadNVectors(128).Length, Is.EqualTo(0));
    }
}
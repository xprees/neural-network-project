using DataLoading;

// using this for always finding the file (even from binary I hope)
string solutionRoot = AppDomain.CurrentDomain.BaseDirectory;
while (!Directory.Exists(Path.Combine(solutionRoot, "data")))
{
    solutionRoot = Directory.GetParent(solutionRoot).FullName;
}
string dataFilePath = Path.Combine(solutionRoot, "data", "fashion_mnist_train_vectors.csv");

DataLoader loader = new DataLoader(dataFilePath);

// reads one line
int[]picture = loader.GetPicBytes();
Console.WriteLine(picture.Length);
// can't read whole file (read by line approach was selected)
picture = loader.GetAllBytes();
Console.WriteLine(picture.Length);
// reads second line
picture = loader.GetPicBytes();
Console.WriteLine(picture.Length); 


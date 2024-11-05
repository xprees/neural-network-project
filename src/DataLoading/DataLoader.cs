namespace DataLoading;

public class DataLoader
{
    private readonly FileStream _fileStream;
    private readonly StreamReader _streamReader;

    private bool? _byRow = null;
    

    public DataLoader(string path)
    {
        _fileStream = new FileStream(path, FileMode.Open, FileAccess.Read);
        _streamReader = new StreamReader(_fileStream);
    }

    // Read one line (one picture) and returns as int[]
    public int[] GetPicBytes()
    {
        if (_byRow == null)
        {
            _byRow = true;
        } else if (_byRow == false)
        {
            return new int[0];
        }

        string line = _streamReader.ReadLine() ?? throw new InvalidOperationException();
        return line.Split(new[] { ',', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => int.Parse(s.Trim()))
            .ToArray();
    }

    // Reads whole file and returns as int[] (size 28*28*rows)
    public int[] GetAllBytes()
    {
        if (_byRow == null)
        {
            _byRow = false;
        } else
        {
            return new int[0];
        }
        string bytes = _streamReader.ReadToEnd();
        
        Close();
        
        return bytes.Split(new[] { ',', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => int.Parse(s.Trim()))
            .ToArray();
    }

    private void Close()
    {
        _streamReader.Close();
        _fileStream.Close();
    }
    
}
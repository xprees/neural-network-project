namespace DataLoading;

public class DataLoader
{
    private const int RowSize = 28;
    private const int BytesInRow = RowSize * RowSize;

    private FileStream _fileStream;
    private BinaryReader _reader;

    public DataLoader(string path)
    {
        _fileStream = new FileStream(path, FileMode.Open, FileAccess.Read);

        if (_fileStream == null)
        {
            throw new IOException("File could not be opened.");
        }
        else
        {
            _reader = new BinaryReader(_fileStream);
        }
    }

    public byte[] GetPicBytes()
    {
        return _reader.ReadBytes(BytesInRow);
    }

    public byte[] GetAllBytes()
    {
        return _reader.ReadBytes((int) _fileStream.Length);
    }

    public void Close()
    {
        _reader.Close();
        _fileStream.Close();
    }
    
}
namespace DataLoading
{
    public class DataLoader : IDisposable
    {
        private readonly StreamReader _streamReader;

        private readonly bool _byRow;

        public DataLoader(string path, bool byRow = true)
        {
            _streamReader = new StreamReader(new FileStream(path, FileMode.Open, FileAccess.Read));
            _byRow = byRow;
        }

        public int[] GetPicBytes()
        {
            if (!_byRow)
            {
                throw new ApplicationException("Reading whole file was specified in constructor");
            }
            string line = _streamReader.ReadLine();
            if (line == null)
            {
                throw new InvalidOperationException("End of file reached or file is empty.");
            }

            return ParseLine(line);
        }

        public int[] GetAllBytes()
        {
            if (_byRow)
            {
                throw new ApplicationException("Reading by row was specified in constructor");
            }

            int[] allLines = ParseLine(_streamReader.ReadToEnd());

            Dispose();
            return allLines.ToArray();
        }

        private static int[] ParseLine(string line)
        {
            try
            {
                return line.Split(new[] { ',', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(s => int.Parse(s.Trim()))
                    .ToArray();
            }
            catch (FormatException ex)
            {
                throw new InvalidDataException("Data format is invalid.", ex);
            }
        }

        public void Dispose()
        {
            _streamReader?.Dispose();
        }
    }
}
using System.Globalization;
using ApplicationException = System.ApplicationException;

namespace DataLoading
{
    public class DataLoader : IDisposable
    {
        private readonly StreamReader _streamReader;

        private readonly bool _byRow;
        
        // Creates DataLoader object
        public DataLoader(string path, bool byRow = true)
        {
            _streamReader = new StreamReader(new FileStream(path, FileMode.Open, FileAccess.Read));
            _byRow = byRow;
        }
        
        // Reads one line of specified CSV file
        public float[] ReadOneVector()
        {
            if (_streamReader.Peek() < 0)
            {
                return [];
            }
            if (!_byRow)
            {
                throw new ApplicationException("Reading whole file was specified in constructor");
            }
            var line = _streamReader.ReadLine();
            if (line == null)
            {
                throw new InvalidOperationException("End of file reached or file is empty.");
            }

            return ParseLine(line)[0];
        }
        

        // Reads batch of n vectors from file
        public float[][] ReadNVectors(int n)
        {
            if (_streamReader.Peek() < 0)
            {
                return [];
            }
            if (!_byRow)
            {
                throw new ApplicationException("Reading whole file was specified in the constructor");
            }
            
            var nLines = new float[n][];
            for (int i = 0; i < n; i++)
            {
                nLines[i] = ReadOneVector();
            }
            
            return nLines;
        }


        // Reads whole specified CSV file
        public float[][] ReadAllVectors()
        {
            if (_streamReader.Peek() < 0)
            {
                return [];
            }
            if (_byRow)
            {
                throw new ApplicationException("Reading by row was specified in constructor");
            }

            var allLines = ParseLine(_streamReader.ReadToEnd());

            return allLines;
        }
        
        // Parse string by \n and comma to float[][] array
        private static float[][] ParseLine(string line)
        {
            try
            {
                return line
                    .Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries) // Split by lines
                    .Select(l => l
                        .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries) // Split by commas in each line
                        .Select(number => float.Parse(number, CultureInfo.InvariantCulture)) // Parse each number to float
                        .ToArray())
                    .ToArray();
            }
            catch (FormatException ex)
            {
                throw new InvalidDataException("Data format is invalid.", ex);
            }
        }

        public void Dispose()
        {
            _streamReader.Dispose();
        }
    }
}
using System.Globalization;

namespace DataLoading
{
    public class DataLoader : IDisposable
    {
        private readonly StreamReader _streamReader;

        private readonly bool _byRow;

        /// <summary>
        /// Creates DataLoader object
        /// </summary>
        /// <param name="path">path to the file</param>
        /// <param name="byRow">true if reading by row, false if reading whole file</param>
        public DataLoader(string path, bool byRow = true)
        {
            _streamReader = new StreamReader(new FileStream(path, FileMode.Open, FileAccess.Read));
            _byRow = byRow;
        }

        /// <summary>
        /// Reads one line of specified CSV file
        /// </summary>
        /// <returns>returns float[][] of values from one line</returns>
        /// <exception cref="ApplicationException">Other read type specified in constructor</exception>
        /// <exception cref="InvalidOperationException">EOF</exception>
        public float[][] ReadOneVector()
        {
            if (_streamReader.Peek() < 0)
            {
                return new float[0][];
            }
            if (!_byRow)
            {
                throw new ApplicationException("Reading whole file was specified in constructor");
            }
            string? line = _streamReader.ReadLine();
            if (line == null)
            {
                throw new InvalidOperationException("End of file reached or file is empty.");
            }

            return ParseLine(line);
        }
        
        /// <summary>
        /// Reads batch of n vectors from file
        /// </summary>
        /// <param name="n">number of rows to be read</param>
        /// <returns>array of n arrays (if not enough rows, rest is null)</returns>
        /// <exception cref="ApplicationException"></exception>
        public float[][] ReadNVectors(int n)
        {
            if (_streamReader.Peek() < 0)
            {
                return new float[0][];
            }
            if (!_byRow)
            {
                throw new ApplicationException("Reading whole file was specified in the constructor");
            }
            
            float [][] nLines = new float[n][];
            for (int i = 0; i < n; i++)
            {
                if (_streamReader.Peek() < 0)
                {
                    return nLines;
                }
                nLines[i] = ReadOneVector()[0];
            }
            
            return nLines;
        }

        /// <summary>
        /// Reads whole specified CSV file
        /// </summary>
        /// <returns>float[][] of all rows from the file</returns>
        /// <exception cref="ApplicationException">Other read type specified in constructor</exception>
        public float[][] ReadAllVectors()
        {
            if (_streamReader.Peek() < 0)
            {
                return new float[0][];
            }
            if (_byRow)
            {
                throw new ApplicationException("Reading by row was specified in constructor");
            }

            float[][] allLines = ParseLine(_streamReader.ReadToEnd());

            return allLines;
        }

        /// <summary>
        /// Parse string by \n and comma to float[][] array
        /// </summary>
        /// <param name="line">string to be parsed</param>
        /// <returns>int[] of all values from line</returns>
        /// <exception cref="InvalidDataException">unexpected data format</exception>
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
            _streamReader?.Dispose();
        }
    }
}
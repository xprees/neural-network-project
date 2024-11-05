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
        /// <returns>returns int[] of values from one line</returns>
        /// <exception cref="ApplicationException">Other read type specified in constructor</exception>
        /// <exception cref="InvalidOperationException">EOF</exception>
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

        /// <summary>
        /// Reads whole specified CSV file
        /// </summary>
        /// <returns>int[] of all numbers from file</returns>
        /// <exception cref="ApplicationException">Other read type specified in constructor</exception>
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

        /// <summary>
        /// Removes commas and \n characters and parse string to int array
        /// </summary>
        /// <param name="line">string to be parsed</param>
        /// <returns>int[] of all values from line</returns>
        /// <exception cref="InvalidDataException">unexpected data format</exception>
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
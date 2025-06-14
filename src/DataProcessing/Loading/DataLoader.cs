﻿using System.Globalization;

namespace DataProcessing.Loading
{
    public class DataLoader(string path) : IDisposable
    {
        private readonly StreamReader _streamReader = new(new FileStream(path, FileMode.Open, FileAccess.Read));

        /// Reads one line of specified CSV file
        public float[] ReadOneVector()
        {
            if (_streamReader.Peek() < 0) return [];

            var line = _streamReader.ReadLine();
            if (line == null) return [];

            return ParseLine(line)[0];
        }


        /// Reads batch of n vectors from file
        public float[][] ReadNVectors(int n)
        {
            if (_streamReader.Peek() < 0) return [];

            var nLines = new float[n][];
            for (var i = 0; i < n; i++)
            {
                nLines[i] = ReadOneVector();
            }

            return nLines;
        }


        /// Reads whole specified CSV file
        public float[][] ReadAllVectors()
        {
            if (_streamReader.Peek() < 0) return [];

            return ParseLine(_streamReader.ReadToEnd());
        }

        /// Parse string by \n and comma to float[][] array
        private static float[][] ParseLine(string line)
        {
            try
            {
                return line
                    .Split("\n", StringSplitOptions.RemoveEmptyEntries) // Split by lines
                    .Select(l => l
                        .Split(",", StringSplitOptions.RemoveEmptyEntries) // Split by commas in each line
                        .Select(number =>
                            float.Parse(number, CultureInfo.InvariantCulture)) // Parse each number to float
                        .ToArray())
                    .ToArray();
            }
            catch (FormatException ex)
            {
                throw new InvalidDataException("Data format is invalid.", ex);
            }
        }

        public void Dispose() => _streamReader.Dispose();
    }
}
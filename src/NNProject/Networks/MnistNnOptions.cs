namespace NNProject.Networks;

public record MnistNnOptions(
    int MaxEpochs = 15,
    int BatchSize = 64,
    float LearningRate = 0.001f,
    float DecayRateOrBeta1 = 0.9f,
    float Beta2 = 0.999f,
    int Seed = 42,
    bool ShuffleData = true
);
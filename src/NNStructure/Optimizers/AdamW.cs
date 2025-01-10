namespace NNStructure.Optimizers;

/// ADAM - Adaptive Moment Estimation optimizer with decoupled weight decay. <a href="https://paperswithcode.com/method/adamw">Papers with code</a>
/// <a href="https://benihime91.github.io/blog/machinelearning/deeplearning/python3.x/tensorflow2.x/2020/10/08/adamW.html">Implementing L2 and AdamW</a>
/// <remarks>Learning rate is recommend around 1e-3 and Beta1 and Beta2 forgetting params typically around 0.9 and 0.999 and weight decay 0.001 - 0.0005</remarks>
public class AdamW(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0.001f)
    : Adam(learningRate, beta1, beta2)
{
    public float WeightDecay { get; set; } = weightDecay;

    public new float UpdateWeight(float weight, float gradient, ref float velocity, ref float squareGradient)
    {
        TimeStep++;

        velocity = Beta1 * velocity + (1 - Beta1) * gradient; // m(t)
        var mHat = velocity / (1 - MathF.Pow(Beta1, TimeStep));

        squareGradient = Beta2 * squareGradient + (1 - Beta2) * gradient * gradient; // v(t)
        var vHat = squareGradient / (1 - MathF.Pow(Beta2, TimeStep));

        var decayedWeight = weight * (1 - LearningRate * WeightDecay);

        return decayedWeight - LearningRate * mHat / (MathF.Sqrt(vHat) + 1e-8f);
    }
}
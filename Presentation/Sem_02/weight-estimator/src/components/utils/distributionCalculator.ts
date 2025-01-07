import {useState} from "react";

type DistributionType = "normal" | "uniform" | "exponential";

interface ModelParams {
  mean?: number;
  stdDev?: number;
  min?: number;
  max?: number;
  rate?: number;
}

export function distriCalculatro(weight: number, stdDev: number) {
  const randomFactor = Math.random() * 2 - 1; // Random value between -1 and 1
  const result = weight + randomFactor * stdDev;
  return result;
}

export function useDistributionCalculator() {
  const [result, setResult] = useState<number | null>(null);

  const calculateDistribution = (
    type: DistributionType,
    params: ModelParams,
    weight: number,
  ): number => {
    let value: number;

    switch (type) {
      case "normal":
        if (params.mean === undefined || params.stdDev === undefined) {
          throw new Error(
            "Mean and standard deviation are required for normal distribution",
          );
        }
        value = normalDistribution(params.mean, params.stdDev);
        break;
      case "uniform":
        if (params.min === undefined || params.max === undefined) {
          throw new Error("Min and max are required for uniform distribution");
        }
        value = uniformDistribution(params.min, params.max);
        break;
      case "exponential":
        if (params.rate === undefined) {
          throw new Error("Rate is required for exponential distribution");
        }
        value = exponentialDistribution(params.rate);
        break;
      default:
        throw new Error("Invalid distribution type");
    }

    return value * weight;
  };

  const normalDistribution = (mean: number, stdDev: number): number => {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return num * stdDev + mean;
  };

  const uniformDistribution = (min: number, max: number): number => {
    return Math.random() * (max - min) + min;
  };

  const exponentialDistribution = (rate: number): number => {
    return -Math.log(1 - Math.random()) / rate;
  };

  const calculate = (
    type: DistributionType,
    params: ModelParams,
    weight: number,
  ) => {
    const calculatedValue = calculateDistribution(type, params, weight);
    setResult(calculatedValue);
    return calculatedValue;
  };

  return {result, calculate};
}

package alphander.network;

import java.util.Arrays;

public class Network 
{	
	float stepBiases = 0.0033f;
	float stepWeights = 0.033f;
	float weightDecay = 0.00001f;
	float random = 1;
	
	public float error;
	public Layer[] layers;
	public Network(int[] size)
	{
		if(size.length < 2) return;
		
		layers = new Layer[size.length - 1];
		for(int i = 0; i < size.length - 1; i++)
			layers[i] = new Layer(size[i], size[i + 1]);
	}
	
	public float[] run(float[] input)
	{
		layers[0].runLayer(input);
		for(int i = 1; i < layers.length; i++)
			layers[i].runLayer(layers[i - 1].output);
		return layers[layers.length - 1].output.clone();
	}
	
	public void train(float[] expected)
	{
		float[] error = new float[expected.length];
		this.error = 0f;
		for(int i = 0; i < error.length; i++)
		{
			error[i] = layers[layers.length - 1].output[i] - expected[i];
			this.error += error[i];
		}
		layers[layers.length - 1].trainLayer(error);
		for(int i = layers.length - 2; i >= 0; i--)
			layers[i].trainLayer(layers[i + 1].error);
	}
	
	public class Layer
	{
		public float[][] weights;
		public float[] biases;
		public float[] error;
		public float[] output;
		public float[] input;
		
		public Layer(int inSize, int outSize)
		{
			 weights = new float[inSize][outSize];
			 biases = new float[outSize];
			 error =  new float[inSize];
			 output = new float[outSize];
			 input = new float[inSize];
			 for(int i = 0; i < weights.length; i++)
				 for(int j = 0; j < weights[i].length; j++)
					 weights[i][j] = (float) (Math.random() - 0.5f) * random;//Setting weights to random value so you can find their gradient
		}
		
		public float[] runLayer(float[] input)
		{
			this.input = input;
			for (int i = 0; i < output.length; i++)
			{
				output[i] = 0;
				for (int j = 0; j < input.length; j++)
					output[i] += input[j] * weights[j][i] + biases[i];
				output[i] = (float) Math.tanh(output[i]);
			}
			return output;
		}
		
		public void trainLayer(float[] frontError)
		{
			Arrays.fill(error, 0f);
			for(int i = 0; i < output.length; i++)
			{
				frontError[i] *= 1 - (output[i] * output[i]);//tanh derivative is 1 - x^2
				biases[i] -= frontError[i] * stepBiases;
	            for (int j = 0; j < input.length; j++)
	            {
	            	error[j] += weights[j][i] * frontError[i];
	            	weights[j][i] -= frontError[i] * input[j] * stepWeights;
	            	weights[j][i] *= 1 - weightDecay;
	            }
			}
		}
	}
}

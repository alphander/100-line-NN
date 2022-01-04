package alphander.network;

public class DemoUse
{
	public static void main(String[] args)
	{
		Network network = new Network(new int[] {2, 16, 16, 2});
		
		for(int i = 0; i < 100_000; i++)
		{
			//Full adder example
			network.run(new float[] {0, 0});
			network.train(new float[] {0, 0});
			
			network.run(new float[] {0, 1});
			network.train(new float[] {0, 1});
			
			network.run(new float[] {1, 0});
			network.train(new float[] {0, 1});
			
			network.run(new float[] {1, 1});
			network.train(new float[] {1, 0});
		}
		
		float[] result = network.run(new float[] {0.f, 0.75f});
	}
}

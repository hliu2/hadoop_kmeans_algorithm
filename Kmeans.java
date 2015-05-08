package k_means;

import java.io.IOException;
import java.util.*;
import java.io.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

@SuppressWarnings("deprecation")
public class KMeansHadoop {
	public static String OUT = "outfile";
	public static String IN = "inputlarger";
	public static String CENTROID_FILE_NAME = "/centroid.txt";
	public static String OUTPUT_FILE_NAME = "/part-00000";
	public static String DATA_FILE_NAME = "/data.txt";
	public static String JOB_NAME = "KMeans";
	public static String SPLITTER = ",";
	public static float[][] centroids = new float[10][2];
	

	
	public static class Map extends MapReduceBase implements 
			Mapper<LongWritable, Text, IntWritable, Text> {
		@Override
		public void configure(JobConf job) {
			try {
				
				Path[] cacheFiles = DistributedCache.getLocalCacheFiles(job);
				if (cacheFiles != null && cacheFiles.length > 0) {
					String line;
					BufferedReader cacheReader = new BufferedReader(
							new FileReader(cacheFiles[0].toString()));
					try {
						int i = 0;
						while ((line = cacheReader.readLine()) != null) {
							String[] temp = line.split(SPLITTER);
							centroids[i][0] = Float.parseFloat(temp[0]);
							centroids[i][1] = Float.parseFloat(temp[1]);
							i++;
							
						}
					} finally {
						cacheReader.close();
					}
				}
			} catch (IOException e) {
				System.err.println("Exception reading DistribtuedCache: " + e);
			}
		}


		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<IntWritable, Text> output,
				Reporter reporter) throws IOException {
			String line = value.toString();
			String[] point = line.split(",");
			float pointx = Float.parseFloat(point[0]);
			float pointy = Float.parseFloat(point[1]);
			float distance = 0;
			float mindistance = 999999999.9f;
			int winnercentroid = -1;
			for (int i = 0; i<10; i++){
				distance = (pointx - centroids[i][0])*(pointx- centroids[i][0])+(pointy-centroids[i][1])*(pointy-centroids[i][1]);
				if(distance < mindistance){
					mindistance = distance;
					winnercentroid = i;
				}
			}
			IntWritable winnerCentroid = new IntWritable(winnercentroid);
			output.collect(winnerCentroid,value);
		}
	}

	public static class Combiner extends MapReduceBase implements
			Reducer<IntWritable, Text, IntWritable, Text> {

		@Override
		public void reduce(IntWritable key, Iterator<Text> values,
				OutputCollector<IntWritable, Text> output, Reporter reporter)
				throws IOException {

			double sumx = 0;
			double sumy = 0;
			int no_elements = 0;
			while (values.hasNext()) {
				String line = values.next().toString();
				String[] record = line.split(",");
				double d1 = Double.parseDouble(record[0]);
				double d2 = Double.parseDouble(record[1]);
				sumx = sumx + d1;
				sumy = sumy + d2;
				++no_elements;
			}
			
			Text output_value = new Text(sumx + "," + sumy + "," + no_elements);

			// Emit new center and point
			output.collect(key, output_value);
		}
	}



	public static class Reduce extends MapReduceBase implements
			Reducer<IntWritable, Text, Text, Text> {

		@Override
		public void reduce(IntWritable key, Iterator<Text> values,
				OutputCollector<Text, Text> output, Reporter reporter)
				throws IOException {
			double newCenterx;
			double newCentery;
			double sumx = 0;
			double sumy = 0;
			int total_count = 0; 
			while (values.hasNext()) {
				String line = values.next().toString();
				String[] record = line.split(",");
				double d1 = Double.parseDouble(record[0]);
				double d2 = Double.parseDouble(record[1]);
                int count = Integer.parseInt(record[2]);
                total_count += count;
				sumx = sumx + d1;
				sumy = sumy + d2;
			}

			// We have new center now
			newCenterx = sumx / total_count;
			newCentery = sumy / total_count;
			
			Text output_value = new Text(newCenterx + "," + newCentery);

			// Emit new center and point
			output.collect(output_value, new Text(" "));
		}
	}

	public static void main(String[] args) throws Exception {
		run(args);
	}

	public static void run(String[] args) throws Exception {
		IN = args[0];
		OUT = args[1];
		String input = IN;
		String output = OUT + System.nanoTime();
		String again_input = output;

		// Reiterating till the convergence
		int iteration = 0;
		boolean isdone = false;
		while (isdone == false) {
			JobConf conf = new JobConf(KMeansHadoop.class);
			if (iteration == 0) {
				Path hdfsPath = new Path(input + CENTROID_FILE_NAME);
				// upload the file to hdfs. Overwrite any existing copy.
				DistributedCache.addCacheFile(hdfsPath.toUri(), conf);
			} else {
				Path hdfsPath = new Path(again_input + OUTPUT_FILE_NAME);
				// upload the file to hdfs. Overwrite any existing copy.
				DistributedCache.addCacheFile(hdfsPath.toUri(), conf);
			}

			conf.setJobName(JOB_NAME);
			conf.set("mapred.textoutputformat.separator","\t");
			conf.setMapOutputKeyClass(IntWritable.class);
			conf.setMapOutputValueClass(Text.class);
			conf.setOutputKeyClass(Text.class);
			conf.setOutputValueClass(Text.class);
			conf.setMapperClass(Map.class);
			conf.setCombinerClass(Combiner.class);
			conf.setReducerClass(Reduce.class);
			conf.setInputFormat(TextInputFormat.class);
			conf.setOutputFormat(TextOutputFormat.class);

			FileInputFormat.setInputPaths(conf,
					new Path(input + DATA_FILE_NAME));
			FileOutputFormat.setOutputPath(conf, new Path(output));

			JobClient.runJob(conf);
			
			Path ofile = new Path(output + OUTPUT_FILE_NAME);
			FileSystem fs = FileSystem.get(new Configuration());
			BufferedReader br = new BufferedReader(new InputStreamReader(
					fs.open(ofile)));
			float[][] centerA = new float[10][2];
			int d = 0;
			String line = br.readLine();
			while (line != null) {
				String[] sp = line.split(",");
				float c1 = Float.parseFloat(sp[0]);
				float c2 = Float.parseFloat(sp[1]);
				centerA[d][0] = c1;
				centerA[d][1] = c2;
				d++;
				line = br.readLine();
			}
			br.close();
			

			String prev;
			if (iteration == 0) {
				prev = input + CENTROID_FILE_NAME;
			} else {
				prev = again_input + OUTPUT_FILE_NAME;
			}
			Path prevfile = new Path(prev);
			FileSystem fs1 = FileSystem.get(new Configuration());
			BufferedReader br1 = new BufferedReader(new InputStreamReader(
					fs1.open(prevfile)));
			float[][] centerB = new float[10][2];
			int d1 = 0;
			String l = br1.readLine();
			while (l != null) {
				String[] sp1 = l.split(",");
				float f1 = Float.parseFloat(sp1[0]);
				float f2 = Float.parseFloat(sp1[1]);
				centerB[d1][0] = f1;
				centerB[d1][1] = f2;
				d1++;
				l = br1.readLine();
			}
			br1.close();
			for(int i = 0; i<10;i++){
				if (centerA[i][0] == centerB[i][0]){
					continue;
				}
				else{
					isdone =false;
				}
			}

			++iteration;
			if(iteration >= 6){
				isdone = true;
			}
			again_input = output;
			output = OUT + System.nanoTime();
		}
	}
}
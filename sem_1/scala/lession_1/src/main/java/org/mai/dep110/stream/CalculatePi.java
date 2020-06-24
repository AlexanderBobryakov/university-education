package org.mai.dep110.stream;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Asus on 10/7/2018.
 */
public class CalculatePi {
    public static void main(String[] args) {
        CalculatePi test = new CalculatePi();
        test.calculatePiCompletable(1000, 1000);
    }

    public void calculatePiCompletable(int parts, int iterations) {
        ExecutorService executor = Executors.newFixedThreadPool(100);

        List<CompletableFuture<Integer>> futures = IntStream
                .range(0, parts)
                .mapToObj(t -> CompletableFuture.supplyAsync(() -> probe(iterations), executor))
                .collect(Collectors.toList());

        Integer inCurcle = futures
                .parallelStream()
                .mapToInt(CompletableFuture::join)
                .reduce((f1, f2) -> f1 + f2)
                .getAsInt();

        executor.shutdown();

        Double pi = (double)inCurcle*4.0d/(double)(parts*iterations);
        System.out.println(pi);
    }

    public Integer probe(int iterations) {
        int inCurcle = 0;
        for (int i = 0; i < iterations; i++) {
            double x = Math.random();
            double y = Math.random();
            if (x * x + y * y <= 1.0d)
                inCurcle++;
        }
        return inCurcle;
    }
}

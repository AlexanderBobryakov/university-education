package org.mai.dep110.stream.example;

import java.awt.geom.Arc2D;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Stream;

public class Main {
    public static void main(String[] args) throws IOException {
        //classic
        Stream<String> strings = Arrays.asList("a", "b", "c").stream();

        //from values
        Stream<String> valuesStream = Stream.of("1", "2", "3");

        //from array
        Stream<Double> doubleStream = Arrays.stream(new Double[] {12.0d, 15.0d});

        //from file
        Stream<String> fromFile = Files.lines(Paths.get("lenses.data"));
        fromFile.forEach(System.out::println);

        //infinite stream from iterate
        Stream<String> iterateExample = Stream.iterate("a", l -> l+"a").limit(3);
        iterateExample.forEach(System.out::println);

        //infinite stream from generate
        Stream generateExample = Stream.generate(Math::random).limit(5);
        generateExample.forEach(System.out::println);
    }
}

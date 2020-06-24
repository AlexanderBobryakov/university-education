package org.mai.dep110.stream.iris;

/**
 * Created by Asus on 10/7/2018.
 */
public class Iris {
    //длина чашелистника
    private double sepalLength;

    //ширина чашелистника
    private double sepalWidth;

    //длина лепестка
    private double petalLength;

    //ширина лепестка
    private double petalWidth;

    //вид
    private String species;

    public Iris(double sepalLength, double sepalWidth, double petalLength, double petalWidth, String species) {
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
        this.species = species;
    }

    public double getSepalLength() {
        return sepalLength;
    }

    public double getSepalWidth() {
        return sepalWidth;
    }

    public double getPetalLength() {
        return petalLength;
    }

    public double getPetalWidth() {
        return petalWidth;
    }

    public String getSpecies() {
        return species;
    }

    static Iris parse(String line) {
        String[] parts = line.split(",");
        Iris result = new Iris(
                Double.parseDouble(parts[0]),
                Double.parseDouble(parts[1]),
                Double.parseDouble(parts[2]),
                Double.parseDouble(parts[3]),
                parts[4]
        );

        return result;
    }

    @Override
    public String toString() {
        return "Iris{" +
                "sepalLength=" + sepalLength +
                ", sepalWidth=" + sepalWidth +
                ", petalLength=" + petalLength +
                ", petalWidth=" + petalWidth +
                ", species='" + species + '\'' +
                '}';
    }

    public Petal classifyByPatel(Iris iris) {
        double patelSquare = iris.getPetalLength()*iris.getPetalLength();
        if(patelSquare < 2.0) {
            return Petal.SMALL;
        } else if(patelSquare < 5.0) {
            return Petal.MEDUIM;
        } else return Petal.LARGE;
    }
}

enum Petal {
    SMALL, MEDUIM, LARGE,
}

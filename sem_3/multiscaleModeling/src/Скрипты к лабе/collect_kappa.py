#!/usr/bin/python3

import os, os.path, csv

def main():
    records = []
    for d in [d for d in os.listdir(".") if os.path.isdir(d)]:
        z = d.split('_')
        R, x, y = float(z[0]), int(z[1]), int(z[2])
        for f in [f for f in os.listdir(d) if f.endswith(".crossplanefilms")]:
            zz = f.split('_')
            T = int(zz[8][:-1])
            with open(d + "/" + f, newline='') as csvfile:
                #reader = csv.reader(f, delimiter=",")
                content = csvfile.readlines()
                for line in content[1:]:
                    row = line.split(",")
                    L, kappa, kappa_bulk = float(row[0]), float(row[1]), float(row[2])
                    records.append({"R":R, "x":x, "y":y, "T":T, "L":L, "kappa":kappa, "kappa_bulk":kappa_bulk})
    with open("kappa.csv", "w") as kappa_csv:
        writer = csv.DictWriter(kappa_csv, fieldnames=["R", "x", "y", "T", "L", "kappa", "kappa_bulk"])
        writer.writeheader()
        for row in records:
            writer.writerow(row)

main()

#!/bin/sh

ALMABTE_PATH="/home/alex/almabte-v1.3.2/build/src"
MATERIALS_PATH="/home/alex/almabte-materials"
#MPI_PROCESSES=12
MPI_PROCESSES=9
Z=24

echo "Started" > log
for R in 0 0.3 0.7; do
        for x in 1 10 15; do
                for y in 1 5 20; do
                        mkdir -p "${R}_${x}_${y}"
                        cd "${R}_${x}_${y}"
                        echo '<superlattice>' > sl.xml
                        echo "  <materials_repository root_directory=\"${MATERIALS_PATH}/\"/>" >> sl.xml
                        echo "  <gridDensity A=\"$Z\" B=\"$Z\" C=\"$Z\"/>" >> sl.xml
                        echo '  <normal na="0" nb="1" nc="1" nqline="501"/>' >> sl.xml
                        echo '  <compound name="GaAs"/>' >> sl.xml
                        echo '  <compound name="InAs"/>' >> sl.xml

SRC="
R = ${R}
phi = 1
n_ML = ${x}

def X(i):
  if i < 1.0:
    return 1.0
  elif 1.0 <= i < n_ML:
    return 1.0 - phi*(1.0 - R**i)
  else:
    return 1.0 - phi*(1.0 - R**n_ML)*R**(i - n_ML)

s = []
ys = []
for i in range(n_ML + ${y}):
	ys.append(X(i))
ys[n_ML+${y}-1] = ys[n_ML+${y}-1] - sum(ys) + ${y}
i_ = n_ML+${y}-1
while ys[i_] < 0 and i_ >= 0:
	ys[i_-1] = ys[i_-1] + ys[i_]
	ys[i_] = 0
	i_ = i_ - 1
for y_ in ys:
        s.append('  <layer mixfraction=\"%f\"/>' % y_)
print('\n'.join(s))"

                        python3 -c "${SRC}" >> sl.xml 

                        echo '  <target directory="."/>' >> sl.xml
                        echo '</superlattice>' >> sl.xml
                        mpirun -np ${MPI_PROCESSES} ${ALMABTE_PATH}/superlattice_builder sl.xml >> ../log

                        SL="$(ls *.h5)"
                        SL="${SL%_${Z}_${Z}_${Z}*}"
                        echo '<crossplanefilmsweep>' > kappa.xml
                        echo '  <H5repository root_directory="."/>' >> kappa.xml
                        echo "  <compound directory=\".\" base=\"${SL}\" gridA=\"${Z}\" gridB=\"${Z}\" gridC=\"${Z}\"/>" >> kappa.xml
                        echo '  <sweep type="log" start="1e-9" stop="1e-4" points="51"/>' >> kappa.xml
                        echo '  <transportAxis x="0" y="0" z="1"/>' >> kappa.xml
                        echo '  <target directory="." file="AUTO"/>' >> kappa.xml
                        echo '</crossplanefilmsweep>' >> kappa.xml
                        for T in 100 150 200 250 300 400 500; do
                                ${ALMABTE_PATH}/kappa_crossplanefilms kappa.xml $T >> ../log
                        done
                        cd ..
                done
        done
done

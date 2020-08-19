./RI_Parallel/ri36 mono gfu ../RI-Datasets/PDBSv1/grouped/singles_all.gff ../RI-Datasets/PDBSv1/grouped/queries_4_all.gff > parallel_out.txt
./RI_Serial/ri36 mono gfu ../RI-Datasets/PDBSv1/grouped/singles_all.gff ../RI-Datasets/PDBSv1/grouped/queries_4_all.gff > serial_out.txt


echo
echo Parallel code info:
echo
sed 1d  parallel_out.txt
echo
echo Sequential code info:
echo
sed 1d serial_out.txt
echo
a=`head -1 parallel_out.txt`
b=`head -1 serial_out.txt`
echo Speedup:
awk "BEGIN {print $b/$a}"
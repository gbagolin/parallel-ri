i=0
while read p; do
i=$(($i+1))
echo "--------------------------------------TEST $i------------------------------------------"
eval "./RI_Parallel/ri36 mono gfu $p > parallel_out.txt"
eval "./RI_Serial/ri36 mono gfu $p > serial_out.txt"

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
echo "---------------------------------------------------------------------------------------"

done < commands.txt
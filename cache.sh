#!/bin/zsh

fmiss="cache_miss.txt"
ftime="cache_time.txt"

tmp="/tmp/likwid.txt"

echo "P\N 2032 2047 2048 2049 2056 2064" > $fmiss
echo "P\AVGTIME" > $ftime

for padding in 0 4 16; do
    make -sB PADDING="-DPADDING=$padding"

    echo -en "$padding " >> $fmiss

    sum=0

    for n in 2032 2047 2048 2049 2056 2064; do
        t=$(sudo likwid-perfctr -C 0-7 -g CACHE -O -o "$tmp" ./rbgs $n $n 8 200 g | awk '/processamento/ {print $5}')
        r=$(grep "Data cache miss ratio" $tmp | tail -n1 | awk -F',' '{print $5}')
        echo -en "$r " >> $fmiss

        sum=$((sum+t))

        echo "$padding $n"
    done

    echo "$padding $((sum/6))" >> $ftime

    echo -en "\n" >> $fmiss
done

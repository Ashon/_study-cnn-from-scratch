#!/bin/bash

FILES='
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
'

OUTPUT_DIR=files

mkdir -p $OUTPUT_DIR

for f in $FILES; do
  output=$OUTPUT_DIR/${f##*/}
  echo "download $f => $output"
  wget -q $f -O $output &
done
wait

echo extract files
gunzip -f $OUTPUT_DIR/*.gz
echo done

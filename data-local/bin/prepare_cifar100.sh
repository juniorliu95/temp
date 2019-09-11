#!/usr/bin/env bash
#
# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # bin

echo "Downloading and unpacking CIFAR-100"
mkdir -p $DIR/../images/cifar/cifar100/by-image/
mkdir -p $DIR/../workdir
mkdir -p $DIR/../labels/cifar100/1000_balanced_labels
mkdir -p $DIR/../labels/cifar100/4000_balanced_labels
python $DIR/unpack_cifar100.py $DIR/../workdir $DIR/../images/cifar/cifar100/by-image/ $DIR/../labels/cifar100

for i in {00..09}
do
# in case i-1.txt excists and the i.txt pollutes i-1.txt
cp $DIR/../labels/cifar100/1000_balanced_labels/$i-1.txt $DIR/../labels/cifar100/1000_balanced_labels/$i.txt
cp $DIR/../labels/cifar100/4000_balanced_labels/$i-1.txt $DIR/../labels/cifar100/4000_balanced_labels/$i.txt

cp $DIR/../labels/cifar100/1000_balanced_labels/$i.txt $DIR/../labels/cifar100/1000_balanced_labels/$i-1.txt
cp $DIR/../labels/cifar100/4000_balanced_labels/$i.txt $DIR/../labels/cifar100/4000_balanced_labels/$i-1.txt
done

rm -rf $DIR/../labels/cifar100/1000_balanced_labels_rand
rm -rf $DIR/../labels/cifar100/4000_balanced_labels_rand
cp -r $DIR/../labels/cifar100/1000_balanced_labels $DIR/../labels/cifar100/1000_balanced_labels_rand
cp -r $DIR/../labels/cifar100/4000_balanced_labels $DIR/../labels/cifar100/4000_balanced_labels_rand

# echo "Linking training set"
# (
#     cd $DIR/../images/cifar/cifar100/by-image/
#     bash $DIR/link_cifar10_train.sh
# )

# echo "Linking validation set"
# (
#     cd $DIR/../images/cifar/cifar100/by-image/
#     bash $DIR/link_cifar10_val.sh
# )

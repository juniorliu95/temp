cd /data1/pytorch-classification-fusion

DATASET=cifar100 # select dataset,cifar10 or cifar100
LABEL_SIZE=1000 # select init labeled dataset,1000 OR 4000

for INIT in {00..05}
# for INIT in 00
do

# init label
LABEL=data-local/labels/$DATASET/${LABEL_SIZE}_balanced_labels_rand/$INIT.txt
ORILABEL=data-local/labels/$DATASET/${LABEL_SIZE}_balanced_labels_rand/$INIT-1.txt
echo $LABEL
cp $ORILABEL $LABEL
echo $INIT "label init..."
rm results/${DATASET}_test_fuse_$INIT.log
echo $INIT "log init..."
rm results/rank_fusion.txt

# start from 1000,choose 1000 each time
for i in {1..10}
# for i in 1
do
echo 'round' $i
# train on L, eval on U, adjust label
python cifar.py \
-a resnet18 \
-d $DATASET \
--labels $LABEL \
--epochs 164 \
--epoch-num 5 \
--select-num 1000 \
--fuse True \
--schedule 81 122 \
--gamma 0.1 \
--wd 1e-4 \
-c checkpoints/$DATASET/resnet-18-fuse/ \
## --epochs 164

cp checkpoints/$DATASET/resnet-18-fuse/log.txt checkpoints/$DATASET/resnet-18-fuse/log_$i.txt

# test
python cifar.py -a resnet18 -d $DATASET --labels $LABEL -e --resume checkpoints/$DATASET/resnet-18-fuse/checkpoint.pth.tar --test-file results/${DATASET}_test_fuse_$INIT.log

done
done

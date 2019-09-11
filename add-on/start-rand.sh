cd /data1/pytorch-classification
for INIT in {00..05}
do

# init label
LABEL=data-local/labels/cifar10/1000_balanced_labels_rand/$INIT.txt
ORILABEL=data-local/labels/cifar10/1000_balanced_labels_rand/$INIT-1.txt
echo $LABEL
cp $ORILABEL $LABEL
echo $INIT "label init..."
rm results/test_rand_$INIT.log
echo $INIT "log init..."

# start from 1000,choose 1000 each time
for i in {1..10}
do
echo 'round' $i
# train on L, eval on U, adjust label
python cifar.py \
-a resnet18 \
--labels $LABEL \
--epochs 164 \
--epoch-num 5 \
--select-num 1000 \
--rand True \
--schedule 81 122 \
--gamma 0.1 \
--wd 1e-4 \
-c checkpoints/cifar10/resnet-18-rand/ \
## --epochs 164

# test
python cifar.py -a resnet18 -e --resume checkpoints/cifar10/resnet-18-rand/checkpoint.pth.tar --test-file results/test_rand_$INIT.log

done
done

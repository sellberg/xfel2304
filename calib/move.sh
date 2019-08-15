#!/usr/bin/env zsh

if [ $# -lt 1 ]
then
	echo Need folder name to move
	exit
fi

dest=/gpfs/exfel/exp/SPB/201901/p002304/usr/Shared/calib/`echo $1|cut -d- -f1`
echo Moving to $dest
[ ! -d $dest ] && mkdir $dest; cp -v $1/*.h5 $dest/
chmod -v a+r $dest/*

cd $dest
cd ..
echo Current directory: $PWD
bname=`basename $dest`
rm latest
ln -v -s $bname latest

dest=/gpfs/exfel/exp/SPB/201901/p002304/scratch/cheetah/calib/agipd/$1
echo Moving to $dest
[ ! -d $dest ] && mkdir $dest; chmod -v a+r $dest
cp -v $1/*.h5 $dest/
chmod -v a+r $dest/*

cd $dest
cd ..
echo Current directory: $PWD
bname=`basename $dest`
rm latest
ln -v -s $bname latest

cd -

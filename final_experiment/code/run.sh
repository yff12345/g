i=0
for model in "MLP" "CNN" "GraphConv" "LR";
do
	for ws in 0.25 0.5 1 1.5 2;
	do
		for ef in "psd" "wav" "raw";
		do
			echo "export CUDA_VISIBLE_DEVICES=$i && nohup python run_experiments.py -m $model -ws $ws -ef $ef &"
			export CUDA_VISIBLE_DEVICES=$i && nohup python run_experiments.py -m $model -ws $ws -ef $ef &
			i=$((i+1))
			if [ $i -gt 3 ]
			then 
				i=0
			fi
		done
	done
done


#export CUDA_VISIBLE_DEVICES=1 && nohup python run_experiments.py -m LR -ws .5 -ef psd &

# All experiments using 1 x 0.5s should sample the same data
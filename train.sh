## print date five times ##
p=1
while [ $p -lt 33 ]; 
do
  x=1
  while [ $x -lt 5 ]; 
  do 
    ./main.py -esp 10 -sg 0 -lr 1 -l2 0.001 -t $x -pf $p -pt $p
    x=$(($x+1))
  done
  echo participant $p >> log.txt
  ./main.py -t 4 -pf $p -pt $p --test >> log.txt
   p=$(($p+1))
done

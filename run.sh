if [ $1 == "S2S" ]; then
	./test.sh model/Seq2Seq/model-77 $2 $3
elif [ $1 == 'RL' ]; then
	./test_RL.sh model/RL/model-56-3000 $2 $3
else
	./test_RL.sh model/RL/model-56-3000 $2 $3
fi
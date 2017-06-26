wget -nc -O model/RL/model-56-3000.data-00000-of-00001 https://www.dropbox.com/s/enggbxeh4m0whg4/model-56-3000.data-00000-of-00001?dl=0
wget -nc -O model/Seq2Seq/model-77.data-00000-of-00001 https://www.dropbox.com/s/ea5pz0jmp5dyrv0/model-77.data-00000-of-00001?dl=0

if [ $1 == "S2S" ]; then
	./test.sh model/Seq2Seq/model-77 $2 $3
elif [ $1 == 'RL' ]; then
	./test_RL.sh model/RL/model-56-3000 $2 $3
else
	./test_RL.sh model/RL/model-56-3000 $2 $3
fi
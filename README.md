# MLDS hw4
an open-domain chatbot

### generate sentence by pre-trained model
```bash
./run.sh <TYPE> <INPUT FILE> <OUTPUT FILE>
```
\<TYPE\> can be one of below:
1. S2S
2. RL
3. BEST

\<INPUT FILE\> is the testing data

you can just use sample_input.txt or sample_input_old.txt in this repository for convenience

\<OUTPUT FILE\> is the output of input file

type any filename you want

### chatbot's results
you can find chatbot's results in tmp/ directory

networks config are described in results' filename

### start training
###### the training config is set up in python/config.py

you can change some training hyper-parameters

you need those libraries
1. python2.7
2. python3
3. tensorflow 1.0.1
4. tensorflow-gpu 1.0.1
5. gensim 1.0.1
6. numpy 1.12.*

#### Step1: download data
we use [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

you need to download it and put it into data/ directory

#### Step2: parse data
```bash
./parse.sh
```

#### Step3: train a Seq2Seq model
```bash
./train.sh
```

#### Step4: test a Seq2Seq model
```bash
./test.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

#### Step5: train a RL model
you can change the training_type parameter in python/config.py

'normal' for seq2seq training, 'pg' for policy gradient

you can first train with 'normal' for some epochs till stable

then change the method to 'pg' to optimize the reward function

```bash
./train_RL.sh
```

*When training with policy gradient*

*you may need a reversed model*

*you can train it by your-self*

*or you can download pre-trained reversed model by*
```bash
./download_reversed.sh
```
*you don't need to change any setting about reversed model if you use pre-trained reversed model*

#### Step6: test a RL model
```bash
./test_RL.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

#### Step7: simulate a dialog
```bash
./simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
\<SIMULATE TYPE\> can be 1 or 2

the number represents # of former sentence(s) the chatbot considers

if you choose 1, chatbot will only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

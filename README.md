# Reinforcement Learning Chatbot
an open-domain chatbot

This chatbot is trained by seq2seq model described in [Sequence to Sequence -- Video to Text](https://arxiv.org/abs/1505.00487)

Then use the reward function described in [Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/abs/1606.01541) to increase the performance.

### chatbot's results
you can find chatbot's results in tmp/ directory

neural networks' configs of those results are described in the filename

### generate sentence by pre-trained model
```bash
./download.sh
```
```bash
./run.sh <TYPE> <INPUT FILE> <OUTPUT FILE>
```
\<TYPE\> can be one of below:
1. S2S
2. RL

\<INPUT FILE\> is the testing data

you can just use data/sample_input.txt or data/sample_input_old.txt in this repository for convenience

\<OUTPUT FILE\> is the output of input file

type any filename you want

### simulate a dialog by pre-trained model
```bash
./download.sh
```
```bash
./simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```

\<SIMULATE TYPE\> can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

if you choose 1, chatbot will only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

\<INPUT FILE\> is the testing data

you can just use data/sample_input.txt or data/sample_input_old.txt in this repository for convenience

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

#### Step4-1: test a Seq2Seq model
```bash
./test.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

#### Step4-2: simulate a dialog
```bash
./simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
\<SIMULATE TYPE\> can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

if you choose 1, chatbot will only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

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
*the reversed model is also trained by cornell movie-dialogs dataset, but with source and target reversed.*

*you don't need to change any setting about reversed model if you use pre-trained reversed model*

#### Step6-1: test a RL model
```bash
./test_RL.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

#### Step6-2: simulate a dialog
```bash
./simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
\<SIMULATE TYPE\> can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

if you choose 1, chatbot will only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

# Reinforcement Learning Chatbot
*an open-domain chatbot*

This chatbot is trained by seq2seq model described in [Sequence to Sequence -- Video to Text](https://arxiv.org/abs/1505.00487)

Then use the reward function described in [Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/abs/1606.01541) to increase the performance.

## chatbot's results
you can find chatbot's results in *tmp/* directory

neural networks' configs of those results are described in the filename

* Example 1
>A: I forgot to get the Coca-Cola.

>B: I got something stuck in the head.

>A: It's all right I guess it's pretty common in the lot of shit.

>B: I mean we've been all this together since the kid.

>A: All the more reason.

>B: It's not him it's his fault and he's blind because of god.

>A: And is a strange story.

* Example 2
>A: Can you do me a favor?

>B: Anything.

>A: I want you to bend a father.

>B: Of course.

>A: I'm so sorry I didn't mean to be rude but.

>B: Don't get it at least it's about as much as I can.

* Example3
>A: I didn't do it! I haven't got a gun!

>B: You liar. You still want to get your money back.

>A: Something wrong with the truth.

>B: I don't want any money just a little I can't take just out of this.

>A: What's wrong.

>B: Listen to me. I'm scared now.

>A: Ah you'll need all the money I need your help.

## generate response by pre-trained model
generate a response from a given request
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

you can just use *data/sample_input.txt* or *data/sample_input_old.txt* in this repository for convenience

or you can create your own testing data (copy the format from abovementioned files)

\<OUTPUT FILE\> is the output of input file

type any filename you want

## simulate a dialog by pre-trained model
generate a dialog, which starts from a given request
```bash
./download.sh
```
```bash
./simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
for \<PATH TO MODEL\>

to generate seq2seq dialog, type model/Seq2Seq/model-77

to generate RL dialog, type model/RL/model-56-3000

\<SIMULATE TYPE\> can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

if you choose 1, chatbot only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

\<INPUT FILE\> is the testing data

you can just use *data/sample_input.txt* or *data/sample_input_old.txt* in this repository for convenience

or you can create your own testing data (copy the format from abovementioned files)

\<OUTPUT FILE\> is the output of input file

type any filename you want

## start training
### Step0: change configs
###### the training config is located in *python/config.py*

you can change some training hyper-parameters, or just keep the original one

### Step1: download data & libraries
we use [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

you need to download it and put it into data/ directory

and you need to download those libraries:
1. python2.7
2. python3
3. tensorflow 1.0.1
4. tensorflow-gpu 1.0.1
5. gensim 1.0.1
6. numpy 1.12.*

### Step2: parse data
```bash
./parse.sh
```

### Step3: train a Seq2Seq model
```bash
./train.sh
```

### Step4-1: test a Seq2Seq model
```bash
./test.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

### Step4-2: simulate a dialog
```bash
./simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
\<SIMULATE TYPE\> can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

if you choose 1, chatbot will only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

### Step5: train a RL model
you can change the training_type parameter in *python/config.py*

'normal' for seq2seq training, 'pg' for policy gradient

you need to first train with 'normal' for some epochs till stable (at least 30 epoches is highly recommended)

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

### Step6-1: test a RL model
```bash
./test_RL.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

### Step6-2: simulate a dialog
```bash
./simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
\<SIMULATE TYPE\> can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

if you choose 1, chatbot will only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

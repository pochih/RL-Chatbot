[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)


## Intro
This is a chatbot trained by __seq2seq__ and __reinforcement learning__.

* __seq2seq__

Seq2seq is a classical model that its input and output are both sequence.

The vanilla seq2seq model is described in NIPS 2014 paper Sequence to [Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), the encoder and the decoder are seperated

My seq2seq model is constructed with 2 LSTMs, similar to the one described in ICCV 2015 paper [Sequence to Sequence -- Video to Text](https://arxiv.org/abs/1505.00487), the encoder and the decoder share weights

* __RL__

After training chatbot with enough epochs, I use one kind of RL techniques called policy gradient, to train chatbot.

By doing this, the chatbot can generate more interesting response, correspond to the reward function.

My reward function is similar to the one described in EMNLP 2016 paper <a href="https://arxiv.org/abs/1606.01541" target="_blank">Deep Reinforcement Learning for Dialogue Generation</a>



## chatbot's results

Let me show you some chatbot's results:

* __Result 1__ (A & B are both a trained chatbot)
>A: Can you do me a favor?

>B: Anything.

>A: I want you to bend a father.

>B: Of course.

>A: I'm so sorry I didn't mean to be rude but.

>B: Don't get it at least it's about as much as I can.

* __Result 2__ (A & B are both a trained chatbot)
>A: I didn't do it! I haven't got a gun!

>B: You liar. You still want to get your money back.

>A: Something wrong with the truth.

>B: I don't want any money just a little I can't take just out of this.

>A: What's wrong.

>B: Listen to me. I'm scared now.

>A: Ah you'll need all the money I need your help.

* __Result 3__ (A & B are both a trained chatbot)
>A: I forgot to get the Coca-Cola.

>B: I got something stuck in the head.

>A: It's all right I guess it's pretty common in the lot of shit.

>B: I mean we've been all this together since the kid.

>A: All the more reason.

>B: It's not him it's his fault and he's blind because of god.

>A: And is a strange story.

you can find more results from chatbot in <a href='tmp' target="_blank">here</a>

neural networks' configs of those results are described in the filename



## simulate a dialog by pre-trained model
Are you amazed at the result of the chatbot? :D

You can generate those amazing dialogs by your own!

Just follow the instructions below:

```bash
./script/download.sh
```
###### (Mention that I use -nc parameter in script/download.sh, it will omit downloading if the file exists

###### So make sure there's no break during the download)
```bash
./script/simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```

- \<PATH TO MODEL\>

to generate seq2seq dialog, type "__model/Seq2Seq/model-77__"

to generate RL dialog, type "__model/RL/model-56-3000__"

- \<SIMULATE TYPE\> 

can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

__if you choose 1, chatbot only considers last sentence__

__if you choose 2, chatbot will consider last two sentences (one from user, and one from chatbot itself)__

- \<INPUT FILE\>

Take a look at <a href='tmp/sample_input_new.txt' target="_blank">example file</a> 

This is the input format of the chatbot, each line is the begin sentence of a dialog.

You can just use the example file for convenience.

- \<OUTPUT FILE\> 

the output file, type any filename you want



## generate response by pre-trained model
If you want chatbot to generate only a single response

Follow the instructions below:

```bash
./script/download.sh
```
###### (Mention that I use -nc parameter in script/download.sh, it will omit downloading if the file exists. So make sure there's no break during the download)
```bash
./script/run.sh <TYPE> <INPUT FILE> <OUTPUT FILE>
```

- \<TYPE\> 

to generate seq2seq response, type "__S2S__"

to generate reinforcement learning response, type "__RL__"

- \<INPUT FILE\> 

Take a look at <a href='tmp/sample_input_new.txt' target="_blank">example file</a>

This is the input format of the chatbot, each line is the begin sentence of a dialog.

You can just use the example file for convenience.

- \<OUTPUT FILE\> 

the output file, type any filename you want



## train chatbot from scratch
I trained my chatbot with python2.7.

If you want to train the chatbot from scratch

You can follow those instructions below:

#### Step0: training configs
Take a look at <a href='python/config.py' target="_blank">config.py</a>, all configs for training is described here.

You can change some training hyper-parameters, or just keep the original ones.

#### Step1: download data & libraries
I use <a href='https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html' target="_blank">Cornell Movie-Dialogs Corpus</a>

You need to download it, unzip it, and __move all .txt files into data/ directory__

Then download some libraries with pip:
```bash
pip install -r requirements.txt
```

#### Step2: parse data 
###### (in this step I use python3)
```bash
./script/parse.sh
```

#### Step3: train a Seq2Seq model
```bash
./script/train.sh
```

#### Step4-1: test a Seq2Seq model
Let's show some results of seq2seq model :)
```bash
./script/test.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

#### Step4-2: simulate a dialog
And show some dialog results from seq2seq model!
```bash
./script/simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
- \<SIMULATE TYPE\> 

can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

if you choose 1, chatbot will only considers user's utterance

if you choose 2, chatbot will considers user's utterance and chatbot's last utterance

#### Step5: train a RL model
you need to change the *training_type* parameter in <a href='python/config.py' target="_blank">config.py</a>

'normal' for seq2seq training, 'pg' for policy gradient

you need to first train with 'normal' for some epochs till stable (at least 30 epoches is highly recommended)

then change the method to 'pg' to optimize the reward function

```bash
./script/train_RL.sh
```

*When training with policy gradient (pg)*

*you may need a reversed model*

*the reversed model is also trained by cornell movie-dialogs dataset, but with source and target reversed.*

*you can download pre-trained reversed model by*
```bash
./script/download_reversed.sh
```

*or you can train it by your-self*

*you don't need to change any setting about reversed model if you use pre-trained reversed model*

#### Step6-1: test a RL model
Let's show some results of RL model, and find the different from seq2seq model :)
```bash
./script/test_RL.sh <PATH TO MODEL> <INPUT FILE> <OUTPUT FILE>
```

#### Step6-2: simulate a dialog
And show some dialog results from RL model!
```bash
./script/simulate.sh <PATH TO MODEL> <SIMULATE TYPE> <INPUT FILE> <OUTPUT FILE>
```
- \<SIMULATE TYPE\> 

can be 1 or 2

the number represents # of former sentence(s) that chatbot considers

__if you choose 1, chatbot only considers last sentence__

__if you choose 2, chatbot will consider last two sentences (one from user, and one from chatbot itself)__


## Author
Po-Chih Huang / [@brianhuang1019](http://brianhuang1019.github.io/)
